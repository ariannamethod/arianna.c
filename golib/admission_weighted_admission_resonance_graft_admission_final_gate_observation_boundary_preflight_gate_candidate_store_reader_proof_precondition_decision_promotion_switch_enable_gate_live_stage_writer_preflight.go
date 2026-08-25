package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport struct {
	Schema                                                                                                                                                                               string `json:"schema"`
	Status                                                                                                                                                                               string `json:"status"`
	Target                                                                                                                                                                               string `json:"target"`
	TargetKind                                                                                                                                                                           string `json:"target_kind"`
	TargetMode                                                                                                                                                                           string `json:"target_mode"`
	Action                                                                                                                                                                               string `json:"action"`
	WriterState                                                                                                                                                                          string `json:"writer_state"`
	WriterAction                                                                                                                                                                         string `json:"writer_action"`
	RollbackState                                                                                                                                                                        string `json:"rollback_state"`
	RollbackAction                                                                                                                                                                       string `json:"rollback_action"`
	StageState                                                                                                                                                                           string `json:"stage_state"`
	StageAction                                                                                                                                                                          string `json:"stage_action"`
	EnableState                                                                                                                                                                          string `json:"enable_state"`
	EnableAction                                                                                                                                                                         string `json:"enable_action"`
	SwitchState                                                                                                                                                                          string `json:"switch_state"`
	SwitchAction                                                                                                                                                                         string `json:"switch_action"`
	Promotion                                                                                                                                                                            string `json:"promotion"`
	LedgerState                                                                                                                                                                          string `json:"ledger_state"`
	LedgerAction                                                                                                                                                                         string `json:"ledger_action"`
	LedgerContract                                                                                                                                                                       string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                                                     string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                                                   string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                                                     string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                                          bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                                                  bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageConsumed             bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageRequired             bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_id"`
	ReceiptShape                                                                                                                                                                         string `json:"receipt_shape"`
	WriterPreflightKind                                                                                                                                                                  string `json:"writer_preflight_kind"`
	WriterPreflightMode                                                                                                                                                                  string `json:"writer_preflight_mode"`
	WriterPreflightStage                                                                                                                                                                 string `json:"writer_preflight_stage"`
	CausalID                                                                                                                                                                             string `json:"causal_id"`
	WriterPreflightHash                                                                                                                                                                  string `json:"writer_preflight_hash"`
	ReadBackHash                                                                                                                                                                         string `json:"read_back_hash"`
	LiveStageVerified                                                                                                                                                                    bool   `json:"live_stage_verified"`
	LiveStageHashVerified                                                                                                                                                                bool   `json:"live_stage_hash_verified"`
	LiveStageReadBackVerified                                                                                                                                                            bool   `json:"live_stage_read_back_verified"`
	EnableGateVerified                                                                                                                                                                   bool   `json:"enable_gate_verified"`
	EnableGateHashVerified                                                                                                                                                               bool   `json:"enable_gate_hash_verified"`
	EnableGateReadBackVerified                                                                                                                                                           bool   `json:"enable_gate_read_back_verified"`
	SwitchVerified                                                                                                                                                                       bool   `json:"switch_verified"`
	SwitchHashVerified                                                                                                                                                                   bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                                                                                                                                               bool   `json:"switch_read_back_verified"`
	PromotionVerified                                                                                                                                                                    bool   `json:"promotion_verified"`
	PromotionHashVerified                                                                                                                                                                bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                                                                                                                                            bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                                                                                                                                     bool   `json:"decision_verified"`
	DecisionHashVerified                                                                                                                                                                 bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                                                                                                                                             bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                                                                                                                                            bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                                                                             bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                                                                         bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                                                                        bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                                                                    bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                                                                bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                                                                  bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                                                        bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                                                    bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                                         bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                                                    bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                                                     bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                                                  bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                                                     bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                                                       bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                                                    bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                                         bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                                                       bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                                                    bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                                                                   bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                                                                 bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                                                               bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                                                                    bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                                                                bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                                                                    bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                                           bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                                                         bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                                                           bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                                            bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                                                  bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                                                 bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                                                bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                                                  bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                                                  bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                                                  bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                                                  bool   `json:"body_mutation_allowed"`
	RequiresWriter                                                                                                                                                                       bool   `json:"requires_writer"`
	WriterReady                                                                                                                                                                          bool   `json:"writer_ready"`
	RollbackRequired                                                                                                                                                                     bool   `json:"rollback_required"`
	RequiresRollback                                                                                                                                                                     bool   `json:"requires_rollback"`
	RollbackReady                                                                                                                                                                        bool   `json:"rollback_ready"`
	ReadOnly                                                                                                                                                                             bool   `json:"read_only"`
	ReplayOnly                                                                                                                                                                           bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                                                                     bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                                                       bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                                         bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                                                     bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                                                 bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                                         bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                                           string `json:"body_target"`
	Passed                                                                                                                                                                               bool   `json:"passed"`
	Reason                                                                                                                                                                               string `json:"reason"`

	SourceSchema                                                                                                                                                                   string `json:"source_schema"`
	SourceStatus                                                                                                                                                                   string `json:"source_status"`
	SourceTarget                                                                                                                                                                   string `json:"source_target"`
	SourceReport                                                                                                                                                                   string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_read_back_hash"`
	SourceStageState                                                                                                                                                               string `json:"source_stage_state"`
	SourceStageAction                                                                                                                                                              string `json:"source_stage_action"`
	SourceLiveStageReceiptShape                                                                                                                                                    string `json:"source_live_stage_receipt_shape"`
	SourceLiveStageKind                                                                                                                                                            string `json:"source_live_stage_kind"`
	SourceLiveStageMode                                                                                                                                                            string `json:"source_live_stage_mode"`
	SourceLiveStageStage                                                                                                                                                           string `json:"source_live_stage_stage"`
	SourceLiveStageLedgerReady                                                                                                                                                     bool   `json:"source_live_stage_ledger_ready"`
	SourceLiveStageLedgerAppendAllowed                                                                                                                                             bool   `json:"source_live_stage_ledger_append_allowed"`
	SourceLiveStageAdmissionRequired                                                                                                                                               bool   `json:"source_live_stage_admission_required"`
	SourceLiveStageShadowOnly                                                                                                                                                      bool   `json:"source_live_stage_shadow_only"`
	SourceLiveStageGraftAllowed                                                                                                                                                    bool   `json:"source_live_stage_graft_allowed"`
	SourceLiveStageDryRunOnly                                                                                                                                                      bool   `json:"source_live_stage_dry_run_only"`
	SourceLiveStageLiveReady                                                                                                                                                       bool   `json:"source_live_stage_live_ready"`
	SourceLiveStageBodyMutationAllowed                                                                                                                                             bool   `json:"source_live_stage_body_mutation_allowed"`
	SourceLiveStageRequiresWriter                                                                                                                                                  bool   `json:"source_live_stage_requires_writer"`
	SourceLiveStageWriterReady                                                                                                                                                     bool   `json:"source_live_stage_writer_ready"`
	SourceLiveStageRollbackRequired                                                                                                                                                bool   `json:"source_live_stage_rollback_required"`
	SourceLiveStageRequiresRollback                                                                                                                                                bool   `json:"source_live_stage_requires_rollback"`
	SourceLiveStageRollbackReady                                                                                                                                                   bool   `json:"source_live_stage_rollback_ready"`
	SourceLiveStageReadOnly                                                                                                                                                        bool   `json:"source_live_stage_read_only"`
	SourceLiveStageReplayOnly                                                                                                                                                      bool   `json:"source_live_stage_replay_only"`
	SourceLiveStageAuthorityGranted                                                                                                                                                bool   `json:"source_live_stage_authority_granted"`
	SourceLiveStageContractsReady                                                                                                                                                  bool   `json:"source_live_stage_contracts_ready"`
	SourceLiveStageWriteAllowed                                                                                                                                                    bool   `json:"source_live_stage_write_allowed"`
	SourceLiveStageAdmissionAllowed                                                                                                                                                bool   `json:"source_live_stage_admission_allowed"`
	SourceLiveStageLiveAdmissionEnabled                                                                                                                                            bool   `json:"source_live_stage_live_admission_enabled"`
	SourceLiveStageMutatesState                                                                                                                                                    bool   `json:"source_live_stage_mutates_state"`
	SourceLiveStageBodyTarget                                                                                                                                                      string `json:"source_live_stage_body_target"`
	SourceLiveStagePassed                                                                                                                                                          bool   `json:"source_live_stage_passed"`
	SourceLiveStageReason                                                                                                                                                          string `json:"source_live_stage_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_read_back_hash"`
	SourceEnableState                                                                                                                                                              string `json:"source_enable_state"`
	SourceEnableAction                                                                                                                                                             string `json:"source_enable_action"`
	SourceEnableGateKind                                                                                                                                                           string `json:"source_enable_gate_kind"`
	SourceEnableGateLedgerAppendAllowed                                                                                                                                            bool   `json:"source_enable_gate_ledger_append_allowed"`
	SourceEnableGateGraftAllowed                                                                                                                                                   bool   `json:"source_enable_gate_graft_allowed"`
	SourceEnableGateWriteAllowed                                                                                                                                                   bool   `json:"source_enable_gate_write_allowed"`
	SourceEnableGateAdmissionAllowed                                                                                                                                               bool   `json:"source_enable_gate_admission_allowed"`
	SourceEnableGateLiveAdmissionEnabled                                                                                                                                           bool   `json:"source_enable_gate_live_admission_enabled"`
	SourceEnableGateBodyMutationAllowed                                                                                                                                            bool   `json:"source_enable_gate_body_mutation_allowed"`
	SourceEnableGateBodyTarget                                                                                                                                                     string `json:"source_enable_gate_body_target"`
	SourceEnableGatePassed                                                                                                                                                         bool   `json:"source_enable_gate_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready"`
	SourceSwitchState                                                                                                                                                              string `json:"source_switch_state"`
	SourceSwitchAction                                                                                                                                                             string `json:"source_switch_action"`
	SourceSwitchKind                                                                                                                                                               string `json:"source_switch_kind"`
	SourceSwitchLedgerAppendAllowed                                                                                                                                                bool   `json:"source_switch_ledger_append_allowed"`
	SourceSwitchGraftAllowed                                                                                                                                                       bool   `json:"source_switch_graft_allowed"`
	SourceSwitchWriteAllowed                                                                                                                                                       bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                                                                                                                                   bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                                                                                                                               bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchBodyMutationAllowed                                                                                                                                                bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchBodyTarget                                                                                                                                                         string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                                                                                                                             bool   `json:"source_switch_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready"`
	SourcePromotion                                                                                                                                                                string `json:"source_promotion"`
	SourcePromotionKind                                                                                                                                                            string `json:"source_promotion_kind"`
	SourcePromotionLedgerAppendAllowed                                                                                                                                             bool   `json:"source_promotion_ledger_append_allowed"`
	SourcePromotionGraftAllowed                                                                                                                                                    bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionWriteAllowed                                                                                                                                                    bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                                                                                                                                bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                                                                                                                            bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionBodyMutationAllowed                                                                                                                                             bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionBodyTarget                                                                                                                                                      string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                                                                                                                          bool   `json:"source_promotion_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	SourceDecision                                                                                                                                                                 string `json:"source_decision"`
	SourceDecisionKind                                                                                                                                                             string `json:"source_decision_kind"`
	SourceDecisionLedgerAppendAllowed                                                                                                                                              bool   `json:"source_decision_ledger_append_allowed"`
	SourceDecisionGraftAllowed                                                                                                                                                     bool   `json:"source_decision_graft_allowed"`
	SourceDecisionWriteAllowed                                                                                                                                                     bool   `json:"source_decision_write_allowed"`
	SourceDecisionLiveAdmissionEnabled                                                                                                                                             bool   `json:"source_decision_live_admission_enabled"`
	SourceDecisionBodyMutationAllowed                                                                                                                                              bool   `json:"source_decision_body_mutation_allowed"`
	SourceDecisionBodyTarget                                                                                                                                                       string `json:"source_decision_body_target"`
	SourceDecisionPassed                                                                                                                                                           bool   `json:"source_decision_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID                                                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady                                              bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                                                                  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                                                                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                                                                      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                                  bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                                                                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                                                                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                                                                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                                                                           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                                                                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                                                              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                                                                           bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                                                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                                                                        bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                                                                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                                                                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceAdmissionRequired                                                                                                                                                        bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                                                               bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                                                               bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                                                           bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                                                                         bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                                                                         bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                                                                 bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                                                               bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_PREFLIGHT_REPORT")
	}
	liveStagePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight output path missing")
	}
	sourceStage, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReportForAssert(liveStagePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReportError(sourceStage, root); err != nil {
		return err
	}
	stage := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight",
		TargetMode:          "closed_writer_preflight_guard_dry_run",
		Action:              "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run",
		WriterState:         "blocked",
		WriterAction:        "reject_blocked_live_stage",
		RollbackState:       "blocked",
		RollbackAction:      "reject_blocked_live_stage",
		StageState:          sourceStage.StageState,
		StageAction:         sourceStage.StageAction,
		EnableState:         sourceStage.EnableState,
		EnableAction:        sourceStage.EnableAction,
		SwitchState:         sourceStage.SwitchState,
		SwitchAction:        sourceStage.SwitchAction,
		Promotion:           sourceStage.Promotion,
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageConsumed:             true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageRequired:             true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight: true,
		ReceiptShape:                 "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_receipt",
		WriterPreflightKind:          "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight",
		WriterPreflightMode:          "closed_live_stage_writer_preflight_guard",
		WriterPreflightStage:         "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_pre_writer_inventory_preflight",
		LiveStageVerified:            true,
		LiveStageHashVerified:        true,
		LiveStageReadBackVerified:    true,
		EnableGateVerified:           sourceStage.EnableGateVerified,
		EnableGateHashVerified:       sourceStage.EnableGateHashVerified,
		EnableGateReadBackVerified:   sourceStage.EnableGateReadBackVerified,
		SwitchVerified:               sourceStage.SwitchVerified,
		SwitchHashVerified:           sourceStage.SwitchHashVerified,
		SwitchReadBackVerified:       sourceStage.SwitchReadBackVerified,
		PromotionVerified:            sourceStage.PromotionVerified,
		PromotionHashVerified:        sourceStage.PromotionHashVerified,
		PromotionReadBackVerified:    sourceStage.PromotionReadBackVerified,
		DecisionVerified:             sourceStage.DecisionVerified,
		DecisionHashVerified:         sourceStage.DecisionHashVerified,
		DecisionReadBackVerified:     sourceStage.DecisionReadBackVerified,
		ProofPreconditionVerified:    sourceStage.ProofPreconditionVerified,
		PreconditionHashVerified:     sourceStage.PreconditionHashVerified,
		PreconditionReadBackVerified: sourceStage.PreconditionReadBackVerified,
		ProofVerified:                sourceStage.ProofVerified,
		ProofHashVerified:            sourceStage.ProofHashVerified,
		ProofReadBackVerified:        sourceStage.ProofReadBackVerified,
		StoreReaderVerified:          sourceStage.StoreReaderVerified,
		StoreVerified:                sourceStage.StoreVerified,
		CandidateVerified:            sourceStage.CandidateVerified,
		GateVerified:                 sourceStage.GateVerified,
		PreflightVerified:            sourceStage.PreflightVerified,
		BoundaryVerified:             sourceStage.BoundaryVerified,
		ObservationVerified:          sourceStage.ObservationVerified,
		ReceiverVerified:             sourceStage.ReceiverVerified,
		IntentVerified:               sourceStage.IntentVerified,
		FinalGateVerified:            sourceStage.FinalGateVerified,
		SealVerified:                 sourceStage.SealVerified,
		PermitVerified:               sourceStage.PermitVerified,
		AuthorityVerified:            sourceStage.AuthorityVerified,
		ReaderHashVerified:           sourceStage.ReaderHashVerified,
		ReaderReplayVerified:         sourceStage.ReaderReplayVerified,
		ReaderReadBackVerified:       sourceStage.ReaderReadBackVerified,
		StoreHashVerified:            sourceStage.StoreHashVerified,
		StoreReadBackVerified:        sourceStage.StoreReadBackVerified,
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
		RequiresWriter:               true,
		WriterReady:                  false,
		RollbackRequired:             true,
		RequiresRollback:             true,
		RollbackReady:                false,
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
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight blocked by blocked live stage; writer and rollback remain absent",

		SourceSchema: sourceStage.Schema, SourceStatus: sourceStage.Status, SourceTarget: sourceStage.Target, SourceReport: liveStagePath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID:       sourceStage.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady:    sourceStage.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID: sourceStage.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash:     sourceStage.LiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack: sourceStage.ReadBackHash,
		SourceStageState: sourceStage.StageState, SourceStageAction: sourceStage.StageAction, SourceLiveStageReceiptShape: sourceStage.ReceiptShape, SourceLiveStageKind: sourceStage.LiveStageKind, SourceLiveStageMode: sourceStage.LiveStageMode, SourceLiveStageStage: sourceStage.LiveStageStage,
		SourceLiveStageLedgerReady: sourceStage.LedgerReady, SourceLiveStageLedgerAppendAllowed: sourceStage.LedgerAppendAllowed,
		SourceLiveStageAdmissionRequired: sourceStage.AdmissionRequired, SourceLiveStageShadowOnly: sourceStage.ShadowOnly, SourceLiveStageGraftAllowed: sourceStage.GraftAllowed, SourceLiveStageDryRunOnly: sourceStage.DryRunOnly, SourceLiveStageLiveReady: sourceStage.LiveReady,
		SourceLiveStageBodyMutationAllowed: sourceStage.BodyMutationAllowed, SourceLiveStageRequiresWriter: sourceStage.RequiresWriter, SourceLiveStageWriterReady: sourceStage.WriterReady, SourceLiveStageRollbackRequired: sourceStage.RollbackRequired, SourceLiveStageRequiresRollback: sourceStage.RequiresRollback, SourceLiveStageRollbackReady: sourceStage.RollbackReady, SourceLiveStageReadOnly: sourceStage.ReadOnly, SourceLiveStageReplayOnly: sourceStage.ReplayOnly,
		SourceLiveStageAuthorityGranted: sourceStage.AuthorityGranted, SourceLiveStageContractsReady: sourceStage.ContractsReady, SourceLiveStageWriteAllowed: sourceStage.WriteAllowed, SourceLiveStageAdmissionAllowed: sourceStage.AdmissionAllowed, SourceLiveStageLiveAdmissionEnabled: sourceStage.LiveAdmissionEnabled, SourceLiveStageMutatesState: sourceStage.MutatesState, SourceLiveStageBodyTarget: sourceStage.BodyTarget, SourceLiveStagePassed: sourceStage.Passed, SourceLiveStageReason: sourceStage.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID:       sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash:     sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack,
		SourceEnableState: sourceStage.SourceEnableState, SourceEnableAction: sourceStage.SourceEnableAction, SourceEnableGateKind: sourceStage.SourceEnableGateKind,
		SourceEnableGateLedgerAppendAllowed: sourceStage.SourceEnableGateLedgerAppendAllowed, SourceEnableGateGraftAllowed: sourceStage.SourceEnableGateGraftAllowed, SourceEnableGateWriteAllowed: sourceStage.SourceEnableGateWriteAllowed, SourceEnableGateAdmissionAllowed: sourceStage.SourceEnableGateAdmissionAllowed, SourceEnableGateLiveAdmissionEnabled: sourceStage.SourceEnableGateLiveAdmissionEnabled, SourceEnableGateBodyMutationAllowed: sourceStage.SourceEnableGateBodyMutationAllowed, SourceEnableGateBodyTarget: sourceStage.SourceEnableGateBodyTarget, SourceEnableGatePassed: sourceStage.SourceEnableGatePassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady,
		SourceSwitchState: sourceStage.SourceSwitchState, SourceSwitchAction: sourceStage.SourceSwitchAction, SourceSwitchKind: sourceStage.SourceSwitchKind,
		SourceSwitchLedgerAppendAllowed: sourceStage.SourceSwitchLedgerAppendAllowed, SourceSwitchGraftAllowed: sourceStage.SourceSwitchGraftAllowed, SourceSwitchWriteAllowed: sourceStage.SourceSwitchWriteAllowed, SourceSwitchAdmissionAllowed: sourceStage.SourceSwitchAdmissionAllowed, SourceSwitchLiveAdmissionEnabled: sourceStage.SourceSwitchLiveAdmissionEnabled, SourceSwitchBodyMutationAllowed: sourceStage.SourceSwitchBodyMutationAllowed, SourceSwitchBodyTarget: sourceStage.SourceSwitchBodyTarget, SourceSwitchPassed: sourceStage.SourceSwitchPassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourcePromotion: sourceStage.SourcePromotion, SourcePromotionKind: sourceStage.SourcePromotionKind,
		SourcePromotionLedgerAppendAllowed: sourceStage.SourcePromotionLedgerAppendAllowed, SourcePromotionGraftAllowed: sourceStage.SourcePromotionGraftAllowed, SourcePromotionWriteAllowed: sourceStage.SourcePromotionWriteAllowed, SourcePromotionAdmissionAllowed: sourceStage.SourcePromotionAdmissionAllowed, SourcePromotionLiveAdmissionEnabled: sourceStage.SourcePromotionLiveAdmissionEnabled, SourcePromotionBodyMutationAllowed: sourceStage.SourcePromotionBodyMutationAllowed, SourcePromotionBodyTarget: sourceStage.SourcePromotionBodyTarget, SourcePromotionPassed: sourceStage.SourcePromotionPassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceDecision: sourceStage.SourceDecision, SourceDecisionKind: sourceStage.SourceDecisionKind,
		SourceDecisionLedgerAppendAllowed: sourceStage.SourceDecisionLedgerAppendAllowed, SourceDecisionGraftAllowed: sourceStage.SourceDecisionGraftAllowed, SourceDecisionWriteAllowed: sourceStage.SourceDecisionWriteAllowed, SourceDecisionLiveAdmissionEnabled: sourceStage.SourceDecisionLiveAdmissionEnabled, SourceDecisionBodyMutationAllowed: sourceStage.SourceDecisionBodyMutationAllowed, SourceDecisionBodyTarget: sourceStage.SourceDecisionBodyTarget, SourceDecisionPassed: sourceStage.SourceDecisionPassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                     sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                  sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                           sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                        sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                                                     sourceStage.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                          sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                   sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                              sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                           sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                 sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                              sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                           sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                                                                      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                                                                      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceAdmissionRequired: sourceStage.SourceAdmissionRequired,
		SourceShadowOnly:        sourceStage.SourceShadowOnly,
		SourceDryRunOnly:        sourceStage.SourceDryRunOnly,
		SourceRequiresWriter:    sourceStage.SourceRequiresWriter,
		SourceRollbackRequired:  sourceStage.SourceRollbackRequired,
		SourceRequiresRollback:  sourceStage.SourceRequiresRollback,
		SourceReadOnly:          sourceStage.SourceReadOnly,
		SourceReplayOnly:        sourceStage.SourceReplayOnly,
	}
	stage.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID(stage)
	stage.WriterPreflightHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash(stage)
	stage.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBackHash(stage)
	stage.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID(stage)
	if stage.CausalID == "" ||
		stage.WriterPreflightHash == "" ||
		stage.ReadBackHash == "" ||
		stage.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID == "" ||
		stage.WriterPreflightHash == stage.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight read-back proof failed")
	}
	raw, err := json.MarshalIndent(stage, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_report=%s\n", outputPath, liveStagePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight")
	}
	if report.TargetMode != "closed_writer_preflight_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight target_mode mismatch: got %q want %q", report.TargetMode, "closed_writer_preflight_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_live_stage")
	}
	if report.RollbackState != "blocked" || report.RollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight rollback state mismatch")
	}
	if report.StageState != "blocked" || report.StageAction != "reject_disabled_enable_gate" || report.EnableState != "disabled" || report.EnableAction != "require_operator_key" || report.SwitchState != "disabled" || report.SwitchAction != "hold_pending_live_admission" || report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight upstream state mismatch")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ledger_append" || report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight receipt_shape mismatch: got %q", report.ReceiptShape)
	}
	if report.WriterPreflightKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight" ||
		report.WriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.WriterPreflightStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_pre_writer_inventory_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight},
		{"live_stage_verified", report.LiveStageVerified},
		{"live_stage_hash_verified", report.LiveStageHashVerified},
		{"live_stage_read_back_verified", report.LiveStageReadBackVerified},
		{"enable_gate_verified", report.EnableGateVerified},
		{"enable_gate_hash_verified", report.EnableGateHashVerified},
		{"enable_gate_read_back_verified", report.EnableGateReadBackVerified},
		{"switch_verified", report.SwitchVerified},
		{"switch_hash_verified", report.SwitchHashVerified},
		{"switch_read_back_verified", report.SwitchReadBackVerified},
		{"promotion_verified", report.PromotionVerified},
		{"promotion_hash_verified", report.PromotionHashVerified},
		{"promotion_read_back_verified", report.PromotionReadBackVerified},
		{"decision_verified", report.DecisionVerified},
		{"decision_hash_verified", report.DecisionHashVerified},
		{"decision_read_back_verified", report.DecisionReadBackVerified},
		{"proof_precondition_verified", report.ProofPreconditionVerified},
		{"precondition_hash_verified", report.PreconditionHashVerified},
		{"precondition_read_back_verified", report.PreconditionReadBackVerified},
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
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady},
		{"source_live_stage_admission_required", report.SourceLiveStageAdmissionRequired},
		{"source_live_stage_shadow_only", report.SourceLiveStageShadowOnly},
		{"source_live_stage_dry_run_only", report.SourceLiveStageDryRunOnly},
		{"source_live_stage_live_ready", report.SourceLiveStageLiveReady},
		{"source_live_stage_requires_writer", report.SourceLiveStageRequiresWriter},
		{"source_live_stage_rollback_required", report.SourceLiveStageRollbackRequired},
		{"source_live_stage_requires_rollback", report.SourceLiveStageRequiresRollback},
		{"source_live_stage_read_only", report.SourceLiveStageReadOnly},
		{"source_live_stage_replay_only", report.SourceLiveStageReplayOnly},
		{"source_live_stage_passed", report.SourceLiveStagePassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady},
		{"source_enable_gate_passed", report.SourceEnableGatePassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady},
		{"source_switch_passed", report.SourceSwitchPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady},
		{"source_promotion_passed", report.SourcePromotionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady},
		{"source_decision_passed", report.SourceDecisionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady},
		{"source_weighted_admission_resonance_graft_admission_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady},
		{"source_weighted_admission_resonance_graft_admission_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady},
		{"source_weighted_admission_resonance_graft_admission_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady},
		{"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"source_weighted_admission_resonance_graft_admission_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady},
		{"source_admission_required", report.SourceAdmissionRequired},
		{"source_shadow_only", report.SourceShadowOnly},
		{"source_dry_run_only", report.SourceDryRunOnly},
		{"source_requires_writer", report.SourceRequiresWriter},
		{"source_rollback_required", report.SourceRollbackRequired},
		{"source_requires_rollback", report.SourceRequiresRollback},
		{"source_read_only", report.SourceReadOnly},
		{"source_replay_only", report.SourceReplayOnly},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight %s not ready", required.name)
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
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"source_live_stage_ledger_ready", report.SourceLiveStageLedgerReady},
		{"source_live_stage_ledger_append_allowed", report.SourceLiveStageLedgerAppendAllowed},
		{"source_live_stage_graft_allowed", report.SourceLiveStageGraftAllowed},
		{"source_live_stage_body_mutation_allowed", report.SourceLiveStageBodyMutationAllowed},
		{"source_live_stage_writer_ready", report.SourceLiveStageWriterReady},
		{"source_live_stage_rollback_ready", report.SourceLiveStageRollbackReady},
		{"source_live_stage_authority_granted", report.SourceLiveStageAuthorityGranted},
		{"source_live_stage_contracts_ready", report.SourceLiveStageContractsReady},
		{"source_live_stage_write_allowed", report.SourceLiveStageWriteAllowed},
		{"source_live_stage_admission_allowed", report.SourceLiveStageAdmissionAllowed},
		{"source_live_stage_live_admission_enabled", report.SourceLiveStageLiveAdmissionEnabled},
		{"source_live_stage_mutates_state", report.SourceLiveStageMutatesState},
		{"source_enable_gate_ledger_append_allowed", report.SourceEnableGateLedgerAppendAllowed},
		{"source_enable_gate_graft_allowed", report.SourceEnableGateGraftAllowed},
		{"source_enable_gate_write_allowed", report.SourceEnableGateWriteAllowed},
		{"source_enable_gate_admission_allowed", report.SourceEnableGateAdmissionAllowed},
		{"source_enable_gate_live_admission_enabled", report.SourceEnableGateLiveAdmissionEnabled},
		{"source_enable_gate_body_mutation_allowed", report.SourceEnableGateBodyMutationAllowed},
		{"source_switch_ledger_append_allowed", report.SourceSwitchLedgerAppendAllowed},
		{"source_switch_graft_allowed", report.SourceSwitchGraftAllowed},
		{"source_switch_write_allowed", report.SourceSwitchWriteAllowed},
		{"source_switch_admission_allowed", report.SourceSwitchAdmissionAllowed},
		{"source_switch_live_admission_enabled", report.SourceSwitchLiveAdmissionEnabled},
		{"source_switch_body_mutation_allowed", report.SourceSwitchBodyMutationAllowed},
		{"source_promotion_ledger_append_allowed", report.SourcePromotionLedgerAppendAllowed},
		{"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed},
		{"source_promotion_write_allowed", report.SourcePromotionWriteAllowed},
		{"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed},
		{"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled},
		{"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed},
		{"source_decision_ledger_append_allowed", report.SourceDecisionLedgerAppendAllowed},
		{"source_decision_graft_allowed", report.SourceDecisionGraftAllowed},
		{"source_decision_write_allowed", report.SourceDecisionWriteAllowed},
		{"source_decision_live_admission_enabled", report.SourceDecisionLiveAdmissionEnabled},
		{"source_decision_body_mutation_allowed", report.SourceDecisionBodyMutationAllowed},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID},
		{"causal_id", report.CausalID},
		{"writer_preflight_hash", report.WriterPreflightHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_blocked_dry_run" || report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight source live stage route mismatch")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_receipt" || report.SourceLiveStageKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage" || report.SourceLiveStageMode != "closed_switch_enable_gate_live_stage_guard" || report.SourceLiveStageStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_pre_writer_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight source live stage shape mismatch")
	}
	if report.SourceStageState != "blocked" || report.SourceStageAction != "reject_disabled_enable_gate" || report.SourceEnableState != "disabled" || report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight source live stage state mismatch")
	}
	if report.SourceEnableGateKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate" || report.SourceSwitchState != "disabled" || report.SourceSwitchAction != "hold_pending_live_admission" || report.SourceSwitchKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" || report.SourcePromotion != "pending_live_admission" || report.SourcePromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" || report.SourceDecision != "shadow_ready" || report.SourceDecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight source chain shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceLiveStageBodyTarget != "none" || report.SourceEnableGateBodyTarget != "none" || report.SourceSwitchBodyTarget != "none" || report.SourcePromotionBodyTarget != "none" || report.SourceDecisionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-id-") || !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-causal-") || !strings.HasPrefix(report.WriterPreflightHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-") || !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-read-") || report.WriterPreflightHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight causal_id mismatch")
	}
	if report.WriterPreflightHash == "" || report.WriterPreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight writer_preflight_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight read_back_hash mismatch")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID(stage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport) string {
	h := hashJSON(map[string]interface{}{
		"source_live_stage_id":             stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		"source_live_stage_read_back_hash": stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack,
		"source_enable_gate_id":            stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		"writer_state":                     stage.WriterState,
		"writer_action":                    stage.WriterAction,
		"rollback_state":                   stage.RollbackState,
		"stage_state":                      stage.StageState,
		"stage_action":                     stage.StageAction,
		"writer_preflight_kind":            stage.WriterPreflightKind,
		"writer_preflight_stage":           stage.WriterPreflightStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash(stage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport) string {
	h := hashJSON(map[string]interface{}{
		"causal_id":                        stage.CausalID,
		"source_live_stage_id":             stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		"source_live_stage_hash":           stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash,
		"source_live_stage_read_back_hash": stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack,
		"writer_state":                     stage.WriterState,
		"writer_action":                    stage.WriterAction,
		"rollback_state":                   stage.RollbackState,
		"rollback_action":                  stage.RollbackAction,
		"stage_state":                      stage.StageState,
		"stage_action":                     stage.StageAction,
		"enable_state":                     stage.EnableState,
		"enable_action":                    stage.EnableAction,
		"switch_state":                     stage.SwitchState,
		"switch_action":                    stage.SwitchAction,
		"promotion":                        stage.Promotion,
		"receipt_shape":                    stage.ReceiptShape,
		"writer_preflight_mode":            stage.WriterPreflightMode,
		"live_stage_verified":              stage.LiveStageVerified,
		"live_stage_hash_verified":         stage.LiveStageHashVerified,
		"live_stage_read_back_verified":    stage.LiveStageReadBackVerified,
		"requires_writer":                  stage.RequiresWriter,
		"writer_ready":                     stage.WriterReady,
		"requires_rollback":                stage.RequiresRollback,
		"rollback_ready":                   stage.RollbackReady,
		"read_only":                        stage.ReadOnly,
		"replay_only":                      stage.ReplayOnly,
		"admission_required":               stage.AdmissionRequired,
		"shadow_only":                      stage.ShadowOnly,
		"dry_run_only":                     stage.DryRunOnly,
		"graft_allowed":                    stage.GraftAllowed,
		"ledger_append_allowed":            stage.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBackHash(stage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport) string {
	h := hashJSON(map[string]interface{}{
		"writer_preflight_hash": stage.WriterPreflightHash,
		"source_live_stage_id":  stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		"writer_state":          stage.WriterState,
		"writer_ready":          stage.WriterReady,
		"rollback_ready":        stage.RollbackReady,
		"body_mutation":         stage.BodyMutationAllowed,
		"live_admission":        stage.LiveAdmissionEnabled,
		"write_allowed":         stage.WriteAllowed,
		"admission_allowed":     stage.AdmissionAllowed,
		"ledger_append_allowed": stage.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID(stage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport) string {
	h := hashJSON(map[string]interface{}{
		"schema":                        stage.Schema,
		"status":                        stage.Status,
		"action":                        stage.Action,
		"writer_state":                  stage.WriterState,
		"writer_action":                 stage.WriterAction,
		"rollback_state":                stage.RollbackState,
		"rollback_action":               stage.RollbackAction,
		"stage_state":                   stage.StageState,
		"stage_action":                  stage.StageAction,
		"enable_state":                  stage.EnableState,
		"enable_action":                 stage.EnableAction,
		"switch_state":                  stage.SwitchState,
		"switch_action":                 stage.SwitchAction,
		"promotion":                     stage.Promotion,
		"source_report":                 stage.SourceReport,
		"source_live_stage_id":          stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		"source_enable_gate_id":         stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		"source_switch_id":              stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		"source_promotion_id":           stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		"source_decision_id":            stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		"source_precondition_id":        stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		"source_proof_id":               stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		"source_reader_id":              stage.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		"causal_id":                     stage.CausalID,
		"writer_preflight_hash":         stage.WriterPreflightHash,
		"read_back_hash":                stage.ReadBackHash,
		"receipt_shape":                 stage.ReceiptShape,
		"writer_preflight_kind":         stage.WriterPreflightKind,
		"writer_preflight_mode":         stage.WriterPreflightMode,
		"writer_preflight_stage":        stage.WriterPreflightStage,
		"body_target":                   stage.BodyTarget,
		"ready":                         stage.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady,
		"live_stage_verified":           stage.LiveStageVerified,
		"live_stage_hash_verified":      stage.LiveStageHashVerified,
		"live_stage_read_back_verified": stage.LiveStageReadBackVerified,
		"requires_writer":               stage.RequiresWriter,
		"writer_ready":                  stage.WriterReady,
		"requires_rollback":             stage.RequiresRollback,
		"rollback_ready":                stage.RollbackReady,
		"admission_required":            stage.AdmissionRequired,
		"shadow_only":                   stage.ShadowOnly,
		"graft_allowed":                 stage.GraftAllowed,
		"dry_run_only":                  stage.DryRunOnly,
		"read_only":                     stage.ReadOnly,
		"replay_only":                   stage.ReplayOnly,
		"live_ready":                    stage.LiveReady,
		"contracts_ready":               stage.ContractsReady,
		"write_allowed":                 stage.WriteAllowed,
		"admission_allowed":             stage.AdmissionAllowed,
		"live_admission_enabled":        stage.LiveAdmissionEnabled,
		"mutates_state":                 stage.MutatesState,
		"ledger_append_allowed":         stage.LedgerAppendAllowed,
		"next_step_blocked_without":     stage.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflight,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight decode failed: %w", err)
	}
	return report, root, nil
}
