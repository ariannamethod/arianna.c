package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventorySchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport struct {
	Schema                                                                                                                                                                                        string `json:"schema"`
	Status                                                                                                                                                                                        string `json:"status"`
	Target                                                                                                                                                                                        string `json:"target"`
	TargetKind                                                                                                                                                                                    string `json:"target_kind"`
	TargetMode                                                                                                                                                                                    string `json:"target_mode"`
	Action                                                                                                                                                                                        string `json:"action"`
	WriterState                                                                                                                                                                                   string `json:"writer_state"`
	WriterAction                                                                                                                                                                                  string `json:"writer_action"`
	RollbackState                                                                                                                                                                                 string `json:"rollback_state"`
	RollbackAction                                                                                                                                                                                string `json:"rollback_action"`
	StageState                                                                                                                                                                                    string `json:"stage_state"`
	StageAction                                                                                                                                                                                   string `json:"stage_action"`
	EnableState                                                                                                                                                                                   string `json:"enable_state"`
	EnableAction                                                                                                                                                                                  string `json:"enable_action"`
	SwitchState                                                                                                                                                                                   string `json:"switch_state"`
	SwitchAction                                                                                                                                                                                  string `json:"switch_action"`
	Promotion                                                                                                                                                                                     string `json:"promotion"`
	InventoryState                                                                                                                                                                                string `json:"inventory_state"`
	InventoryAction                                                                                                                                                                               string `json:"inventory_action"`
	WriterContract                                                                                                                                                                                string `json:"writer_contract"`
	RollbackContract                                                                                                                                                                              string `json:"rollback_contract"`
	AdmissionLedgerContract                                                                                                                                                                       string `json:"admission_ledger_contract"`
	WriterContractPresent                                                                                                                                                                         bool   `json:"writer_contract_present"`
	RollbackContractPresent                                                                                                                                                                       bool   `json:"rollback_contract_present"`
	LedgerContractPresent                                                                                                                                                                         bool   `json:"ledger_contract_present"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady          bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightConsumed       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightRequired       bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory          bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID             string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_id"`
	ReceiptShape                                                                                                                                                                                  string `json:"receipt_shape"`
	WriterInventoryKind                                                                                                                                                                           string `json:"writer_inventory_kind"`
	WriterInventoryMode                                                                                                                                                                           string `json:"writer_inventory_mode"`
	WriterInventoryStage                                                                                                                                                                          string `json:"writer_inventory_stage"`
	CausalID                                                                                                                                                                                      string `json:"causal_id"`
	WriterInventoryHash                                                                                                                                                                           string `json:"writer_inventory_hash"`
	ReadBackHash                                                                                                                                                                                  string `json:"read_back_hash"`
	WriterPreflightVerified                                                                                                                                                                       bool   `json:"writer_preflight_verified"`
	WriterPreflightHashVerified                                                                                                                                                                   bool   `json:"writer_preflight_hash_verified"`
	WriterPreflightReadBackVerified                                                                                                                                                               bool   `json:"writer_preflight_read_back_verified"`
	LiveStageVerified                                                                                                                                                                             bool   `json:"live_stage_verified"`
	LiveStageHashVerified                                                                                                                                                                         bool   `json:"live_stage_hash_verified"`
	LiveStageReadBackVerified                                                                                                                                                                     bool   `json:"live_stage_read_back_verified"`
	EnableGateVerified                                                                                                                                                                            bool   `json:"enable_gate_verified"`
	EnableGateHashVerified                                                                                                                                                                        bool   `json:"enable_gate_hash_verified"`
	EnableGateReadBackVerified                                                                                                                                                                    bool   `json:"enable_gate_read_back_verified"`
	SwitchVerified                                                                                                                                                                                bool   `json:"switch_verified"`
	SwitchHashVerified                                                                                                                                                                            bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                                                                                                                                                        bool   `json:"switch_read_back_verified"`
	PromotionVerified                                                                                                                                                                             bool   `json:"promotion_verified"`
	PromotionHashVerified                                                                                                                                                                         bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                                                                                                                                                     bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                                                                                                                                              bool   `json:"decision_verified"`
	DecisionHashVerified                                                                                                                                                                          bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                                                                                                                                                      bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                                                                                                                                                     bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                                                                                      bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                                                                                  bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                                                                                 bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                                                                             bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                                                                         bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                                                                           bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                                                                 bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                                                             bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                                                  bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                                                             bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                                                              bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                                                           bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                                                              bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                                                                bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                                                             bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                                                  bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                                                                bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                                                             bool   `json:"authority_verified"`
	AdmissionRequired                                                                                                                                                                             bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                                                    bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                                                                  bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                                                                    bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                                                     bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                                                           bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                                                          bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                                                         bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                                                           bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                                                           bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                                                           bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                                                           bool   `json:"body_mutation_allowed"`
	RequiresWriter                                                                                                                                                                                bool   `json:"requires_writer"`
	WriterReady                                                                                                                                                                                   bool   `json:"writer_ready"`
	RollbackRequired                                                                                                                                                                              bool   `json:"rollback_required"`
	RequiresRollback                                                                                                                                                                              bool   `json:"requires_rollback"`
	RollbackReady                                                                                                                                                                                 bool   `json:"rollback_ready"`
	ReadOnly                                                                                                                                                                                      bool   `json:"read_only"`
	ReplayOnly                                                                                                                                                                                    bool   `json:"replay_only"`
	SourceSchema                                                                                                                                                                                  string `json:"source_schema"`
	SourceStatus                                                                                                                                                                                  string `json:"source_status"`
	SourceTarget                                                                                                                                                                                  string `json:"source_target"`
	SourceReport                                                                                                                                                                                  string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_read_back_hash"`
	SourceWriterPreflightReceiptShape                                                                                                                                                             string `json:"source_writer_preflight_receipt_shape"`
	SourceWriterPreflightKind                                                                                                                                                                     string `json:"source_writer_preflight_kind"`
	SourceWriterPreflightMode                                                                                                                                                                     string `json:"source_writer_preflight_mode"`
	SourceWriterPreflightStage                                                                                                                                                                    string `json:"source_writer_preflight_stage"`
	SourceWriterPreflightWriterState                                                                                                                                                              string `json:"source_writer_preflight_writer_state"`
	SourceWriterPreflightWriterAction                                                                                                                                                             string `json:"source_writer_preflight_writer_action"`
	SourceWriterPreflightRollbackState                                                                                                                                                            string `json:"source_writer_preflight_rollback_state"`
	SourceWriterPreflightRollbackAction                                                                                                                                                           string `json:"source_writer_preflight_rollback_action"`
	SourceWriterPreflightAdmissionRequired                                                                                                                                                        bool   `json:"source_writer_preflight_admission_required"`
	SourceWriterPreflightShadowOnly                                                                                                                                                               bool   `json:"source_writer_preflight_shadow_only"`
	SourceWriterPreflightGraftAllowed                                                                                                                                                             bool   `json:"source_writer_preflight_graft_allowed"`
	SourceWriterPreflightDryRunOnly                                                                                                                                                               bool   `json:"source_writer_preflight_dry_run_only"`
	SourceWriterPreflightLiveReady                                                                                                                                                                bool   `json:"source_writer_preflight_live_ready"`
	SourceWriterPreflightRawDreamTextAllowed                                                                                                                                                      bool   `json:"source_writer_preflight_raw_dream_text_allowed"`
	SourceWriterPreflightRawDreamTextObserved                                                                                                                                                     bool   `json:"source_writer_preflight_raw_dream_text_observed"`
	SourceWriterPreflightRawDreamTextForwarded                                                                                                                                                    bool   `json:"source_writer_preflight_raw_dream_text_forwarded"`
	SourceWriterPreflightJanusSurfaceAllowed                                                                                                                                                      bool   `json:"source_writer_preflight_janus_surface_allowed"`
	SourceWriterPreflightCoocLearningAllowed                                                                                                                                                      bool   `json:"source_writer_preflight_cooc_learning_allowed"`
	SourceWriterPreflightDeltaHarvestAllowed                                                                                                                                                      bool   `json:"source_writer_preflight_delta_harvest_allowed"`
	SourceWriterPreflightBodyMutationAllowed                                                                                                                                                      bool   `json:"source_writer_preflight_body_mutation_allowed"`
	SourceWriterPreflightRequiresWriter                                                                                                                                                           bool   `json:"source_writer_preflight_requires_writer"`
	SourceWriterPreflightWriterReady                                                                                                                                                              bool   `json:"source_writer_preflight_writer_ready"`
	SourceWriterPreflightRollbackRequired                                                                                                                                                         bool   `json:"source_writer_preflight_rollback_required"`
	SourceWriterPreflightRequiresRollback                                                                                                                                                         bool   `json:"source_writer_preflight_requires_rollback"`
	SourceWriterPreflightRollbackReady                                                                                                                                                            bool   `json:"source_writer_preflight_rollback_ready"`
	SourceWriterPreflightReadOnly                                                                                                                                                                 bool   `json:"source_writer_preflight_read_only"`
	SourceWriterPreflightReplayOnly                                                                                                                                                               bool   `json:"source_writer_preflight_replay_only"`
	SourceWriterPreflightWriteAllowed                                                                                                                                                             bool   `json:"source_writer_preflight_write_allowed"`
	SourceWriterPreflightAdmissionAllowed                                                                                                                                                         bool   `json:"source_writer_preflight_admission_allowed"`
	SourceWriterPreflightLiveAdmissionEnabled                                                                                                                                                     bool   `json:"source_writer_preflight_live_admission_enabled"`
	SourceWriterPreflightMutatesState                                                                                                                                                             bool   `json:"source_writer_preflight_mutates_state"`
	SourceWriterPreflightBodyTarget                                                                                                                                                               string `json:"source_writer_preflight_body_target"`
	SourceWriterPreflightPassed                                                                                                                                                                   bool   `json:"source_writer_preflight_passed"`
	SourceWriterPreflightReason                                                                                                                                                                   string `json:"source_writer_preflight_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID                      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash                    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_read_back_hash"`
	SourceStageState                                                                                                                                                                              string `json:"source_stage_state"`
	SourceStageAction                                                                                                                                                                             string `json:"source_stage_action"`
	SourceLiveStageReceiptShape                                                                                                                                                                   string `json:"source_live_stage_receipt_shape"`
	SourceLiveStageKind                                                                                                                                                                           string `json:"source_live_stage_kind"`
	SourceLiveStageMode                                                                                                                                                                           string `json:"source_live_stage_mode"`
	SourceLiveStageStage                                                                                                                                                                          string `json:"source_live_stage_stage"`
	SourceLiveStageAdmissionRequired                                                                                                                                                              bool   `json:"source_live_stage_admission_required"`
	SourceLiveStageShadowOnly                                                                                                                                                                     bool   `json:"source_live_stage_shadow_only"`
	SourceLiveStageGraftAllowed                                                                                                                                                                   bool   `json:"source_live_stage_graft_allowed"`
	SourceLiveStageDryRunOnly                                                                                                                                                                     bool   `json:"source_live_stage_dry_run_only"`
	SourceLiveStageLiveReady                                                                                                                                                                      bool   `json:"source_live_stage_live_ready"`
	SourceLiveStageRawDreamTextAllowed                                                                                                                                                            bool   `json:"source_live_stage_raw_dream_text_allowed"`
	SourceLiveStageRawDreamTextObserved                                                                                                                                                           bool   `json:"source_live_stage_raw_dream_text_observed"`
	SourceLiveStageRawDreamTextForwarded                                                                                                                                                          bool   `json:"source_live_stage_raw_dream_text_forwarded"`
	SourceLiveStageJanusSurfaceAllowed                                                                                                                                                            bool   `json:"source_live_stage_janus_surface_allowed"`
	SourceLiveStageCoocLearningAllowed                                                                                                                                                            bool   `json:"source_live_stage_cooc_learning_allowed"`
	SourceLiveStageDeltaHarvestAllowed                                                                                                                                                            bool   `json:"source_live_stage_delta_harvest_allowed"`
	SourceLiveStageBodyMutationAllowed                                                                                                                                                            bool   `json:"source_live_stage_body_mutation_allowed"`
	SourceLiveStageRequiresWriter                                                                                                                                                                 bool   `json:"source_live_stage_requires_writer"`
	SourceLiveStageWriterReady                                                                                                                                                                    bool   `json:"source_live_stage_writer_ready"`
	SourceLiveStageRollbackRequired                                                                                                                                                               bool   `json:"source_live_stage_rollback_required"`
	SourceLiveStageRequiresRollback                                                                                                                                                               bool   `json:"source_live_stage_requires_rollback"`
	SourceLiveStageRollbackReady                                                                                                                                                                  bool   `json:"source_live_stage_rollback_ready"`
	SourceLiveStageReadOnly                                                                                                                                                                       bool   `json:"source_live_stage_read_only"`
	SourceLiveStageReplayOnly                                                                                                                                                                     bool   `json:"source_live_stage_replay_only"`
	SourceLiveStageWriteAllowed                                                                                                                                                                   bool   `json:"source_live_stage_write_allowed"`
	SourceLiveStageAdmissionAllowed                                                                                                                                                               bool   `json:"source_live_stage_admission_allowed"`
	SourceLiveStageLiveAdmissionEnabled                                                                                                                                                           bool   `json:"source_live_stage_live_admission_enabled"`
	SourceLiveStageMutatesState                                                                                                                                                                   bool   `json:"source_live_stage_mutates_state"`
	SourceLiveStageBodyTarget                                                                                                                                                                     string `json:"source_live_stage_body_target"`
	SourceLiveStagePassed                                                                                                                                                                         bool   `json:"source_live_stage_passed"`
	SourceLiveStageReason                                                                                                                                                                         string `json:"source_live_stage_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID                               string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_read_back_hash"`
	SourceEnableState                                                                                                                                                                             string `json:"source_enable_state"`
	SourceEnableAction                                                                                                                                                                            string `json:"source_enable_action"`
	SourceEnableGateReceiptShape                                                                                                                                                                  string `json:"source_enable_gate_receipt_shape"`
	SourceEnableGateKind                                                                                                                                                                          string `json:"source_enable_gate_kind"`
	SourceEnableGateMode                                                                                                                                                                          string `json:"source_enable_gate_mode"`
	SourceEnableGateStage                                                                                                                                                                         string `json:"source_enable_gate_stage"`
	SourceEnableGateAdmissionRequired                                                                                                                                                             bool   `json:"source_enable_gate_admission_required"`
	SourceEnableGateShadowOnly                                                                                                                                                                    bool   `json:"source_enable_gate_shadow_only"`
	SourceEnableGateGraftAllowed                                                                                                                                                                  bool   `json:"source_enable_gate_graft_allowed"`
	SourceEnableGateDryRunOnly                                                                                                                                                                    bool   `json:"source_enable_gate_dry_run_only"`
	SourceEnableGateLiveReady                                                                                                                                                                     bool   `json:"source_enable_gate_live_ready"`
	SourceEnableGateRawDreamTextAllowed                                                                                                                                                           bool   `json:"source_enable_gate_raw_dream_text_allowed"`
	SourceEnableGateRawDreamTextObserved                                                                                                                                                          bool   `json:"source_enable_gate_raw_dream_text_observed"`
	SourceEnableGateRawDreamTextForwarded                                                                                                                                                         bool   `json:"source_enable_gate_raw_dream_text_forwarded"`
	SourceEnableGateJanusSurfaceAllowed                                                                                                                                                           bool   `json:"source_enable_gate_janus_surface_allowed"`
	SourceEnableGateCoocLearningAllowed                                                                                                                                                           bool   `json:"source_enable_gate_cooc_learning_allowed"`
	SourceEnableGateDeltaHarvestAllowed                                                                                                                                                           bool   `json:"source_enable_gate_delta_harvest_allowed"`
	SourceEnableGateBodyMutationAllowed                                                                                                                                                           bool   `json:"source_enable_gate_body_mutation_allowed"`
	SourceEnableGateRollbackRequired                                                                                                                                                              bool   `json:"source_enable_gate_rollback_required"`
	SourceEnableGateReadOnly                                                                                                                                                                      bool   `json:"source_enable_gate_read_only"`
	SourceEnableGateReplayOnly                                                                                                                                                                    bool   `json:"source_enable_gate_replay_only"`
	SourceEnableGateWriteAllowed                                                                                                                                                                  bool   `json:"source_enable_gate_write_allowed"`
	SourceEnableGateAdmissionAllowed                                                                                                                                                              bool   `json:"source_enable_gate_admission_allowed"`
	SourceEnableGateLiveAdmissionEnabled                                                                                                                                                          bool   `json:"source_enable_gate_live_admission_enabled"`
	SourceEnableGateMutatesState                                                                                                                                                                  bool   `json:"source_enable_gate_mutates_state"`
	SourceEnableGateBodyTarget                                                                                                                                                                    string `json:"source_enable_gate_body_target"`
	SourceEnableGatePassed                                                                                                                                                                        bool   `json:"source_enable_gate_passed"`
	SourceEnableGateReason                                                                                                                                                                        string `json:"source_enable_gate_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID                                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash                                       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack                                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_read_back_hash"`
	SourceSwitchState                                                                                                                                                                             string `json:"source_switch_state"`
	SourceSwitchAction                                                                                                                                                                            string `json:"source_switch_action"`
	SourceSwitchReceiptShape                                                                                                                                                                      string `json:"source_switch_receipt_shape"`
	SourceSwitchKind                                                                                                                                                                              string `json:"source_switch_kind"`
	SourceSwitchMode                                                                                                                                                                              string `json:"source_switch_mode"`
	SourceSwitchStage                                                                                                                                                                             string `json:"source_switch_stage"`
	SourceSwitchAdmissionRequired                                                                                                                                                                 bool   `json:"source_switch_admission_required"`
	SourceSwitchShadowOnly                                                                                                                                                                        bool   `json:"source_switch_shadow_only"`
	SourceSwitchGraftAllowed                                                                                                                                                                      bool   `json:"source_switch_graft_allowed"`
	SourceSwitchDryRunOnly                                                                                                                                                                        bool   `json:"source_switch_dry_run_only"`
	SourceSwitchLiveReady                                                                                                                                                                         bool   `json:"source_switch_live_ready"`
	SourceSwitchRawDreamTextAllowed                                                                                                                                                               bool   `json:"source_switch_raw_dream_text_allowed"`
	SourceSwitchRawDreamTextObserved                                                                                                                                                              bool   `json:"source_switch_raw_dream_text_observed"`
	SourceSwitchRawDreamTextForwarded                                                                                                                                                             bool   `json:"source_switch_raw_dream_text_forwarded"`
	SourceSwitchJanusSurfaceAllowed                                                                                                                                                               bool   `json:"source_switch_janus_surface_allowed"`
	SourceSwitchCoocLearningAllowed                                                                                                                                                               bool   `json:"source_switch_cooc_learning_allowed"`
	SourceSwitchDeltaHarvestAllowed                                                                                                                                                               bool   `json:"source_switch_delta_harvest_allowed"`
	SourceSwitchBodyMutationAllowed                                                                                                                                                               bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchRollbackRequired                                                                                                                                                                  bool   `json:"source_switch_rollback_required"`
	SourceSwitchReadOnly                                                                                                                                                                          bool   `json:"source_switch_read_only"`
	SourceSwitchReplayOnly                                                                                                                                                                        bool   `json:"source_switch_replay_only"`
	SourceSwitchWriteAllowed                                                                                                                                                                      bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                                                                                                                                                  bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                                                                                                                                              bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchMutatesState                                                                                                                                                                      bool   `json:"source_switch_mutates_state"`
	SourceSwitchBodyTarget                                                                                                                                                                        string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                                                                                                                                            bool   `json:"source_switch_passed"`
	SourceSwitchReason                                                                                                                                                                            string `json:"source_switch_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID                                               string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady                                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_read_back_hash"`
	SourcePromotion                                                                                                                                                                               string `json:"source_promotion"`
	SourcePromotionAction                                                                                                                                                                         string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                                                                                                                                                   string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                                                                                                                                           string `json:"source_promotion_kind"`
	SourcePromotionMode                                                                                                                                                                           string `json:"source_promotion_mode"`
	SourcePromotionStage                                                                                                                                                                          string `json:"source_promotion_stage"`
	SourcePromotionAdmissionRequired                                                                                                                                                              bool   `json:"source_promotion_admission_required"`
	SourcePromotionShadowOnly                                                                                                                                                                     bool   `json:"source_promotion_shadow_only"`
	SourcePromotionGraftAllowed                                                                                                                                                                   bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionDryRunOnly                                                                                                                                                                     bool   `json:"source_promotion_dry_run_only"`
	SourcePromotionLiveReady                                                                                                                                                                      bool   `json:"source_promotion_live_ready"`
	SourcePromotionRawDreamTextAllowed                                                                                                                                                            bool   `json:"source_promotion_raw_dream_text_allowed"`
	SourcePromotionRawDreamTextObserved                                                                                                                                                           bool   `json:"source_promotion_raw_dream_text_observed"`
	SourcePromotionRawDreamTextForwarded                                                                                                                                                          bool   `json:"source_promotion_raw_dream_text_forwarded"`
	SourcePromotionJanusSurfaceAllowed                                                                                                                                                            bool   `json:"source_promotion_janus_surface_allowed"`
	SourcePromotionCoocLearningAllowed                                                                                                                                                            bool   `json:"source_promotion_cooc_learning_allowed"`
	SourcePromotionDeltaHarvestAllowed                                                                                                                                                            bool   `json:"source_promotion_delta_harvest_allowed"`
	SourcePromotionBodyMutationAllowed                                                                                                                                                            bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionRollbackRequired                                                                                                                                                               bool   `json:"source_promotion_rollback_required"`
	SourcePromotionReadOnly                                                                                                                                                                       bool   `json:"source_promotion_read_only"`
	SourcePromotionReplayOnly                                                                                                                                                                     bool   `json:"source_promotion_replay_only"`
	SourcePromotionWriteAllowed                                                                                                                                                                   bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                                                                                                                                               bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                                                                                                                                           bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionMutatesState                                                                                                                                                                   bool   `json:"source_promotion_mutates_state"`
	SourcePromotionBodyTarget                                                                                                                                                                     string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                                                                                                                                         bool   `json:"source_promotion_passed"`
	SourcePromotionReason                                                                                                                                                                         string `json:"source_promotion_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID                                                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID                                                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady                                                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                                                                            string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                                                                                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                                                                              bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                                                                                       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                                                                                    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                                                                            string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                                                                                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                                                                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                                                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                                                                                  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                                                                                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                                                                                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                                                                             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                                                                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                                                                                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                                                                                       bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                                                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityConsumed                                                                                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_authority_consumed"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityRequired                                                                                                                               bool   `json:"source_weighted_admission_resonance_graft_admission_authority_required"`
	BodySmokeWeighted                                                                                                                                                                             bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                                                                                                                                              bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                                                                                                                                           bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                                                                                                                                                  bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                                                                                                                                                       bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                                                                                                                                                        bool   `json:"source_authority_granted"`
	AuthorityGranted                                                                                                                                                                              bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                                                                bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                                                  bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                                                              bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                                                          bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                                                  bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                                                    string `json:"body_target"`
	Passed                                                                                                                                                                                        bool   `json:"passed"`
	Reason                                                                                                                                                                                        string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT")
	}
	writerPreflightPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory output path missing")
	}
	sourcePreflight, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportForAssert(writerPreflightPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReportError(sourcePreflight, root); err != nil {
		return err
	}
	inventory := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport{
		Schema:                  admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventorySchema,
		Status:                  "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_blocked_dry_run",
		Target:                  "live_route_admission_next_step",
		TargetKind:              "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory",
		TargetMode:              "closed_writer_inventory_guard_dry_run",
		Action:                  "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run",
		WriterState:             "blocked",
		WriterAction:            "reject_blocked_writer_preflight",
		RollbackState:           "blocked",
		RollbackAction:          "reject_blocked_writer_preflight",
		StageState:              sourcePreflight.StageState,
		StageAction:             sourcePreflight.StageAction,
		EnableState:             sourcePreflight.EnableState,
		EnableAction:            sourcePreflight.EnableAction,
		SwitchState:             sourcePreflight.SwitchState,
		SwitchAction:            sourcePreflight.SwitchAction,
		Promotion:               sourcePreflight.Promotion,
		InventoryState:          "blocked",
		InventoryAction:         "reject_blocked_writer_preflight",
		WriterContract:          "none",
		RollbackContract:        "none",
		AdmissionLedgerContract: "none",
		ReceiptShape:            "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_receipt",
		WriterInventoryKind:     "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory",
		WriterInventoryMode:     "closed_writer_preflight_inventory_guard",
		WriterInventoryStage:    "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_pre_writer_contract_inventory",
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady:    true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory:    true,
		WriterPreflightVerified:         true,
		WriterPreflightHashVerified:     true,
		WriterPreflightReadBackVerified: true,
		LiveStageVerified:               sourcePreflight.LiveStageVerified,
		LiveStageHashVerified:           sourcePreflight.LiveStageHashVerified,
		LiveStageReadBackVerified:       sourcePreflight.LiveStageReadBackVerified,
		EnableGateVerified:              sourcePreflight.EnableGateVerified,
		EnableGateHashVerified:          sourcePreflight.EnableGateHashVerified,
		EnableGateReadBackVerified:      sourcePreflight.EnableGateReadBackVerified,
		SwitchVerified:                  sourcePreflight.SwitchVerified,
		SwitchHashVerified:              sourcePreflight.SwitchHashVerified,
		SwitchReadBackVerified:          sourcePreflight.SwitchReadBackVerified,
		PromotionVerified:               sourcePreflight.PromotionVerified,
		PromotionHashVerified:           sourcePreflight.PromotionHashVerified,
		PromotionReadBackVerified:       sourcePreflight.PromotionReadBackVerified,
		DecisionVerified:                sourcePreflight.DecisionVerified,
		DecisionHashVerified:            sourcePreflight.DecisionHashVerified,
		DecisionReadBackVerified:        sourcePreflight.DecisionReadBackVerified,
		ProofPreconditionVerified:       sourcePreflight.ProofPreconditionVerified,
		PreconditionHashVerified:        sourcePreflight.PreconditionHashVerified,
		PreconditionReadBackVerified:    sourcePreflight.PreconditionReadBackVerified,
		ProofVerified:                   sourcePreflight.ProofVerified,
		ProofHashVerified:               sourcePreflight.ProofHashVerified,
		ProofReadBackVerified:           sourcePreflight.ProofReadBackVerified,
		StoreReaderVerified:             sourcePreflight.StoreReaderVerified,
		StoreVerified:                   sourcePreflight.StoreVerified,
		CandidateVerified:               sourcePreflight.CandidateVerified,
		GateVerified:                    sourcePreflight.GateVerified,
		PreflightVerified:               sourcePreflight.PreflightVerified,
		BoundaryVerified:                sourcePreflight.BoundaryVerified,
		ObservationVerified:             sourcePreflight.ObservationVerified,
		ReceiverVerified:                sourcePreflight.ReceiverVerified,
		IntentVerified:                  sourcePreflight.IntentVerified,
		FinalGateVerified:               sourcePreflight.FinalGateVerified,
		SealVerified:                    sourcePreflight.SealVerified,
		PermitVerified:                  sourcePreflight.PermitVerified,
		AuthorityVerified:               sourcePreflight.AuthorityVerified,
		AdmissionRequired:               true,
		ShadowOnly:                      true,
		GraftAllowed:                    false,
		DryRunOnly:                      true,
		LiveReady:                       true,
		RawDreamTextAllowed:             false,
		RawDreamTextObserved:            false,
		RawDreamTextForwarded:           false,
		JanusSurfaceAllowed:             false,
		CoocLearningAllowed:             false,
		DeltaHarvestAllowed:             false,
		BodyMutationAllowed:             false,
		RequiresWriter:                  true,
		WriterReady:                     false,
		RollbackRequired:                true,
		RequiresRollback:                true,
		RollbackReady:                   false,
		ReadOnly:                        true,
		ReplayOnly:                      true,
		SourceSchema:                    sourcePreflight.Schema,
		SourceStatus:                    sourcePreflight.Status,
		SourceTarget:                    sourcePreflight.Target,
		SourceReport:                    writerPreflightPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID:       sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady:    sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID: sourcePreflight.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash:     sourcePreflight.WriterPreflightHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack: sourcePreflight.ReadBackHash,
		SourceWriterPreflightReceiptShape:          sourcePreflight.ReceiptShape,
		SourceWriterPreflightKind:                  sourcePreflight.WriterPreflightKind,
		SourceWriterPreflightMode:                  sourcePreflight.WriterPreflightMode,
		SourceWriterPreflightStage:                 sourcePreflight.WriterPreflightStage,
		SourceWriterPreflightWriterState:           sourcePreflight.WriterState,
		SourceWriterPreflightWriterAction:          sourcePreflight.WriterAction,
		SourceWriterPreflightRollbackState:         sourcePreflight.RollbackState,
		SourceWriterPreflightRollbackAction:        sourcePreflight.RollbackAction,
		SourceWriterPreflightAdmissionRequired:     sourcePreflight.AdmissionRequired,
		SourceWriterPreflightShadowOnly:            sourcePreflight.ShadowOnly,
		SourceWriterPreflightGraftAllowed:          sourcePreflight.GraftAllowed,
		SourceWriterPreflightDryRunOnly:            sourcePreflight.DryRunOnly,
		SourceWriterPreflightLiveReady:             sourcePreflight.LiveReady,
		SourceWriterPreflightRawDreamTextAllowed:   sourcePreflight.RawDreamTextAllowed,
		SourceWriterPreflightRawDreamTextObserved:  sourcePreflight.RawDreamTextObserved,
		SourceWriterPreflightRawDreamTextForwarded: sourcePreflight.RawDreamTextForwarded,
		SourceWriterPreflightJanusSurfaceAllowed:   sourcePreflight.JanusSurfaceAllowed,
		SourceWriterPreflightCoocLearningAllowed:   sourcePreflight.CoocLearningAllowed,
		SourceWriterPreflightDeltaHarvestAllowed:   sourcePreflight.DeltaHarvestAllowed,
		SourceWriterPreflightBodyMutationAllowed:   sourcePreflight.BodyMutationAllowed,
		SourceWriterPreflightRequiresWriter:        sourcePreflight.RequiresWriter,
		SourceWriterPreflightWriterReady:           sourcePreflight.WriterReady,
		SourceWriterPreflightRollbackRequired:      sourcePreflight.RollbackRequired,
		SourceWriterPreflightRequiresRollback:      sourcePreflight.RequiresRollback,
		SourceWriterPreflightRollbackReady:         sourcePreflight.RollbackReady,
		SourceWriterPreflightReadOnly:              sourcePreflight.ReadOnly,
		SourceWriterPreflightReplayOnly:            sourcePreflight.ReplayOnly,
		SourceWriterPreflightWriteAllowed:          sourcePreflight.WriteAllowed,
		SourceWriterPreflightAdmissionAllowed:      sourcePreflight.AdmissionAllowed,
		SourceWriterPreflightLiveAdmissionEnabled:  sourcePreflight.LiveAdmissionEnabled,
		SourceWriterPreflightMutatesState:          sourcePreflight.MutatesState,
		SourceWriterPreflightBodyTarget:            sourcePreflight.BodyTarget,
		SourceWriterPreflightPassed:                sourcePreflight.Passed,
		SourceWriterPreflightReason:                sourcePreflight.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady:    sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID: sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash:     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack: sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack,
		SourceStageState:                     sourcePreflight.SourceStageState,
		SourceStageAction:                    sourcePreflight.SourceStageAction,
		SourceLiveStageReceiptShape:          sourcePreflight.SourceLiveStageReceiptShape,
		SourceLiveStageKind:                  sourcePreflight.SourceLiveStageKind,
		SourceLiveStageMode:                  sourcePreflight.SourceLiveStageMode,
		SourceLiveStageStage:                 sourcePreflight.SourceLiveStageStage,
		SourceLiveStageAdmissionRequired:     sourcePreflight.SourceLiveStageAdmissionRequired,
		SourceLiveStageShadowOnly:            sourcePreflight.SourceLiveStageShadowOnly,
		SourceLiveStageGraftAllowed:          sourcePreflight.SourceLiveStageGraftAllowed,
		SourceLiveStageDryRunOnly:            sourcePreflight.SourceLiveStageDryRunOnly,
		SourceLiveStageLiveReady:             sourcePreflight.SourceLiveStageLiveReady,
		SourceLiveStageRawDreamTextAllowed:   false,
		SourceLiveStageRawDreamTextObserved:  false,
		SourceLiveStageRawDreamTextForwarded: false,
		SourceLiveStageJanusSurfaceAllowed:   false,
		SourceLiveStageCoocLearningAllowed:   false,
		SourceLiveStageDeltaHarvestAllowed:   false,
		SourceLiveStageBodyMutationAllowed:   sourcePreflight.SourceLiveStageBodyMutationAllowed,
		SourceLiveStageRequiresWriter:        sourcePreflight.SourceLiveStageRequiresWriter,
		SourceLiveStageWriterReady:           sourcePreflight.SourceLiveStageWriterReady,
		SourceLiveStageRollbackRequired:      sourcePreflight.SourceLiveStageRollbackRequired,
		SourceLiveStageRequiresRollback:      sourcePreflight.SourceLiveStageRequiresRollback,
		SourceLiveStageRollbackReady:         sourcePreflight.SourceLiveStageRollbackReady,
		SourceLiveStageReadOnly:              sourcePreflight.SourceLiveStageReadOnly,
		SourceLiveStageReplayOnly:            sourcePreflight.SourceLiveStageReplayOnly,
		SourceLiveStageWriteAllowed:          sourcePreflight.SourceLiveStageWriteAllowed,
		SourceLiveStageAdmissionAllowed:      sourcePreflight.SourceLiveStageAdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:  sourcePreflight.SourceLiveStageLiveAdmissionEnabled,
		SourceLiveStageMutatesState:          sourcePreflight.SourceLiveStageMutatesState,
		SourceLiveStageBodyTarget:            sourcePreflight.SourceLiveStageBodyTarget,
		SourceLiveStagePassed:                sourcePreflight.SourceLiveStagePassed,
		SourceLiveStageReason:                sourcePreflight.SourceLiveStageReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady:    sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID: "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash:     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack: sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack,
		SourceEnableState:                     sourcePreflight.SourceEnableState,
		SourceEnableAction:                    sourcePreflight.SourceEnableAction,
		SourceEnableGateReceiptShape:          "",
		SourceEnableGateKind:                  sourcePreflight.SourceEnableGateKind,
		SourceEnableGateMode:                  "",
		SourceEnableGateStage:                 "",
		SourceEnableGateAdmissionRequired:     false,
		SourceEnableGateShadowOnly:            false,
		SourceEnableGateGraftAllowed:          sourcePreflight.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:            false,
		SourceEnableGateLiveReady:             false,
		SourceEnableGateRawDreamTextAllowed:   false,
		SourceEnableGateRawDreamTextObserved:  false,
		SourceEnableGateRawDreamTextForwarded: false,
		SourceEnableGateJanusSurfaceAllowed:   false,
		SourceEnableGateCoocLearningAllowed:   false,
		SourceEnableGateDeltaHarvestAllowed:   false,
		SourceEnableGateBodyMutationAllowed:   sourcePreflight.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:      false,
		SourceEnableGateReadOnly:              false,
		SourceEnableGateReplayOnly:            false,
		SourceEnableGateWriteAllowed:          sourcePreflight.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:      sourcePreflight.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:  sourcePreflight.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:          false,
		SourceEnableGateBodyTarget:            sourcePreflight.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                sourcePreflight.SourceEnableGatePassed,
		SourceEnableGateReason:                "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady:    sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID: "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash:     "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack: "",
		SourceSwitchState:                 sourcePreflight.SourceSwitchState,
		SourceSwitchAction:                sourcePreflight.SourceSwitchAction,
		SourceSwitchReceiptShape:          "",
		SourceSwitchKind:                  sourcePreflight.SourceSwitchKind,
		SourceSwitchMode:                  "",
		SourceSwitchStage:                 "",
		SourceSwitchAdmissionRequired:     false,
		SourceSwitchShadowOnly:            false,
		SourceSwitchGraftAllowed:          sourcePreflight.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:            false,
		SourceSwitchLiveReady:             false,
		SourceSwitchRawDreamTextAllowed:   false,
		SourceSwitchRawDreamTextObserved:  false,
		SourceSwitchRawDreamTextForwarded: false,
		SourceSwitchJanusSurfaceAllowed:   false,
		SourceSwitchCoocLearningAllowed:   false,
		SourceSwitchDeltaHarvestAllowed:   false,
		SourceSwitchBodyMutationAllowed:   sourcePreflight.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:      false,
		SourceSwitchReadOnly:              false,
		SourceSwitchReplayOnly:            false,
		SourceSwitchWriteAllowed:          sourcePreflight.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:      sourcePreflight.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:  sourcePreflight.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:          false,
		SourceSwitchBodyTarget:            sourcePreflight.SourceSwitchBodyTarget,
		SourceSwitchPassed:                sourcePreflight.SourceSwitchPassed,
		SourceSwitchReason:                "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady:    sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID: "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash:     "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack: "",
		SourcePromotion:                      sourcePreflight.SourcePromotion,
		SourcePromotionAction:                "",
		SourcePromotionReceiptShape:          "",
		SourcePromotionKind:                  sourcePreflight.SourcePromotionKind,
		SourcePromotionMode:                  "",
		SourcePromotionStage:                 "",
		SourcePromotionAdmissionRequired:     false,
		SourcePromotionShadowOnly:            false,
		SourcePromotionGraftAllowed:          sourcePreflight.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:            false,
		SourcePromotionLiveReady:             false,
		SourcePromotionRawDreamTextAllowed:   false,
		SourcePromotionRawDreamTextObserved:  false,
		SourcePromotionRawDreamTextForwarded: false,
		SourcePromotionJanusSurfaceAllowed:   false,
		SourcePromotionCoocLearningAllowed:   false,
		SourcePromotionDeltaHarvestAllowed:   false,
		SourcePromotionBodyMutationAllowed:   sourcePreflight.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:      false,
		SourcePromotionReadOnly:              false,
		SourcePromotionReplayOnly:            false,
		SourcePromotionWriteAllowed:          sourcePreflight.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:      sourcePreflight.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:  sourcePreflight.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:          false,
		SourcePromotionBodyTarget:            sourcePreflight.SourcePromotionBodyTarget,
		SourcePromotionPassed:                sourcePreflight.SourcePromotionPassed,
		SourcePromotionReason:                "",
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:    sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady: sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:            sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady:         sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                        sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:                     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                             sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                                   sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                                sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                        sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                                     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                                 sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                                     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                                  sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                           sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                                      sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                                   sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                         sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                                      sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                        sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                                   sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                                 sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityConsumed:                                                                           false,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityRequired:                                                                           false,
		BodySmokeWeighted:            false,
		NanoDirectRunner:             false,
		NanoDirectFinalGate:          false,
		ResonanceGraftAdmissionProof: false,
		BoundaryReportFullChain:      false,
		SourceAuthorityGranted:       false,
		AuthorityGranted:             false,
		ContractsReady:               false,
		WriteAllowed:                 false,
		AdmissionAllowed:             false,
		LiveAdmissionEnabled:         false,
		MutatesState:                 false,
		BodyTarget:                   "none",
		Passed:                       true,
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent",
	}
	inventory.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID(inventory)
	inventory.WriterInventoryHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash(inventory)
	inventory.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBackHash(inventory)
	inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID(inventory)
	if inventory.CausalID == "" ||
		inventory.WriterInventoryHash == "" ||
		inventory.ReadBackHash == "" ||
		inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID == "" ||
		inventory.WriterInventoryHash == inventory.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory read-back proof failed")
	}
	raw, err := json.MarshalIndent(inventory, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_report=%s\n", outputPath, writerPreflightPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventorySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventorySchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory")
	}
	if report.TargetMode != "closed_writer_inventory_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory target_mode mismatch: got %q want %q", report.TargetMode, "closed_writer_inventory_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_writer_preflight")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_writer_preflight")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.InventoryState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory inventory_state mismatch: got %q want %q", report.InventoryState, "blocked")
	}
	if report.InventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory inventory_action mismatch: got %q want %q", report.InventoryAction, "reject_blocked_writer_preflight")
	}
	if report.WriterContract != "none" || report.RollbackContract != "none" || report.AdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory contracts unexpectedly named")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_receipt")
	}
	if report.WriterInventoryKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory" ||
		report.WriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		report.WriterInventoryStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_pre_writer_contract_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory},
		{"writer_preflight_verified", report.WriterPreflightVerified},
		{"writer_preflight_hash_verified", report.WriterPreflightHashVerified},
		{"writer_preflight_read_back_verified", report.WriterPreflightReadBackVerified},
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
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady},
		{"source_writer_preflight_admission_required", report.SourceWriterPreflightAdmissionRequired},
		{"source_writer_preflight_shadow_only", report.SourceWriterPreflightShadowOnly},
		{"source_writer_preflight_dry_run_only", report.SourceWriterPreflightDryRunOnly},
		{"source_writer_preflight_live_ready", report.SourceWriterPreflightLiveReady},
		{"source_writer_preflight_requires_writer", report.SourceWriterPreflightRequiresWriter},
		{"source_writer_preflight_rollback_required", report.SourceWriterPreflightRollbackRequired},
		{"source_writer_preflight_requires_rollback", report.SourceWriterPreflightRequiresRollback},
		{"source_writer_preflight_read_only", report.SourceWriterPreflightReadOnly},
		{"source_writer_preflight_replay_only", report.SourceWriterPreflightReplayOnly},
		{"source_writer_preflight_passed", report.SourceWriterPreflightPassed},
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
		{"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
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
		{"writer_contract_present", report.WriterContractPresent},
		{"rollback_contract_present", report.RollbackContractPresent},
		{"ledger_contract_present", report.LedgerContractPresent},
		{"source_writer_preflight_graft_allowed", report.SourceWriterPreflightGraftAllowed},
		{"source_writer_preflight_raw_dream_text_allowed", report.SourceWriterPreflightRawDreamTextAllowed},
		{"source_writer_preflight_raw_dream_text_observed", report.SourceWriterPreflightRawDreamTextObserved},
		{"source_writer_preflight_raw_dream_text_forwarded", report.SourceWriterPreflightRawDreamTextForwarded},
		{"source_writer_preflight_janus_surface_allowed", report.SourceWriterPreflightJanusSurfaceAllowed},
		{"source_writer_preflight_cooc_learning_allowed", report.SourceWriterPreflightCoocLearningAllowed},
		{"source_writer_preflight_delta_harvest_allowed", report.SourceWriterPreflightDeltaHarvestAllowed},
		{"source_writer_preflight_body_mutation_allowed", report.SourceWriterPreflightBodyMutationAllowed},
		{"source_writer_preflight_writer_ready", report.SourceWriterPreflightWriterReady},
		{"source_writer_preflight_rollback_ready", report.SourceWriterPreflightRollbackReady},
		{"source_writer_preflight_write_allowed", report.SourceWriterPreflightWriteAllowed},
		{"source_writer_preflight_admission_allowed", report.SourceWriterPreflightAdmissionAllowed},
		{"source_writer_preflight_live_admission_enabled", report.SourceWriterPreflightLiveAdmissionEnabled},
		{"source_writer_preflight_mutates_state", report.SourceWriterPreflightMutatesState},
		{"source_live_stage_graft_allowed", report.SourceLiveStageGraftAllowed},
		{"source_live_stage_raw_dream_text_allowed", report.SourceLiveStageRawDreamTextAllowed},
		{"source_live_stage_raw_dream_text_observed", report.SourceLiveStageRawDreamTextObserved},
		{"source_live_stage_raw_dream_text_forwarded", report.SourceLiveStageRawDreamTextForwarded},
		{"source_live_stage_janus_surface_allowed", report.SourceLiveStageJanusSurfaceAllowed},
		{"source_live_stage_cooc_learning_allowed", report.SourceLiveStageCoocLearningAllowed},
		{"source_live_stage_delta_harvest_allowed", report.SourceLiveStageDeltaHarvestAllowed},
		{"source_live_stage_body_mutation_allowed", report.SourceLiveStageBodyMutationAllowed},
		{"source_live_stage_writer_ready", report.SourceLiveStageWriterReady},
		{"source_live_stage_rollback_ready", report.SourceLiveStageRollbackReady},
		{"source_live_stage_write_allowed", report.SourceLiveStageWriteAllowed},
		{"source_live_stage_admission_allowed", report.SourceLiveStageAdmissionAllowed},
		{"source_live_stage_live_admission_enabled", report.SourceLiveStageLiveAdmissionEnabled},
		{"source_live_stage_mutates_state", report.SourceLiveStageMutatesState},
		{"source_enable_gate_graft_allowed", report.SourceEnableGateGraftAllowed},
		{"source_enable_gate_raw_dream_text_allowed", report.SourceEnableGateRawDreamTextAllowed},
		{"source_enable_gate_raw_dream_text_observed", report.SourceEnableGateRawDreamTextObserved},
		{"source_enable_gate_raw_dream_text_forwarded", report.SourceEnableGateRawDreamTextForwarded},
		{"source_enable_gate_janus_surface_allowed", report.SourceEnableGateJanusSurfaceAllowed},
		{"source_enable_gate_cooc_learning_allowed", report.SourceEnableGateCoocLearningAllowed},
		{"source_enable_gate_delta_harvest_allowed", report.SourceEnableGateDeltaHarvestAllowed},
		{"source_enable_gate_body_mutation_allowed", report.SourceEnableGateBodyMutationAllowed},
		{"source_enable_gate_write_allowed", report.SourceEnableGateWriteAllowed},
		{"source_enable_gate_admission_allowed", report.SourceEnableGateAdmissionAllowed},
		{"source_enable_gate_live_admission_enabled", report.SourceEnableGateLiveAdmissionEnabled},
		{"source_enable_gate_mutates_state", report.SourceEnableGateMutatesState},
		{"source_switch_graft_allowed", report.SourceSwitchGraftAllowed},
		{"source_switch_raw_dream_text_allowed", report.SourceSwitchRawDreamTextAllowed},
		{"source_switch_raw_dream_text_observed", report.SourceSwitchRawDreamTextObserved},
		{"source_switch_raw_dream_text_forwarded", report.SourceSwitchRawDreamTextForwarded},
		{"source_switch_janus_surface_allowed", report.SourceSwitchJanusSurfaceAllowed},
		{"source_switch_cooc_learning_allowed", report.SourceSwitchCoocLearningAllowed},
		{"source_switch_delta_harvest_allowed", report.SourceSwitchDeltaHarvestAllowed},
		{"source_switch_body_mutation_allowed", report.SourceSwitchBodyMutationAllowed},
		{"source_switch_write_allowed", report.SourceSwitchWriteAllowed},
		{"source_switch_admission_allowed", report.SourceSwitchAdmissionAllowed},
		{"source_switch_live_admission_enabled", report.SourceSwitchLiveAdmissionEnabled},
		{"source_switch_mutates_state", report.SourceSwitchMutatesState},
		{"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed},
		{"source_promotion_raw_dream_text_allowed", report.SourcePromotionRawDreamTextAllowed},
		{"source_promotion_raw_dream_text_observed", report.SourcePromotionRawDreamTextObserved},
		{"source_promotion_raw_dream_text_forwarded", report.SourcePromotionRawDreamTextForwarded},
		{"source_promotion_janus_surface_allowed", report.SourcePromotionJanusSurfaceAllowed},
		{"source_promotion_cooc_learning_allowed", report.SourcePromotionCoocLearningAllowed},
		{"source_promotion_delta_harvest_allowed", report.SourcePromotionDeltaHarvestAllowed},
		{"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed},
		{"source_promotion_write_allowed", report.SourcePromotionWriteAllowed},
		{"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed},
		{"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled},
		{"source_promotion_mutates_state", report.SourcePromotionMutatesState},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID},
		{"causal_id", report.CausalID},
		{"writer_inventory_hash", report.WriterInventoryHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceWriterPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_receipt" ||
		report.SourceWriterPreflightKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight" ||
		report.SourceWriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.SourceWriterPreflightStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_pre_writer_inventory_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source writer preflight shape mismatch")
	}
	if report.SourceWriterPreflightWriterState != "blocked" ||
		report.SourceWriterPreflightWriterAction != "reject_blocked_live_stage" ||
		report.SourceWriterPreflightRollbackState != "blocked" ||
		report.SourceWriterPreflightRollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source writer preflight state mismatch")
	}
	if report.SourceWriterPreflightBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_writer_preflight_body_target mismatch: got %q want %q", report.SourceWriterPreflightBodyTarget, "none")
	}
	if report.SourceWriterPreflightReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_writer_preflight_reason mismatch: got %q", report.SourceWriterPreflightReason)
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage" ||
		report.SourceLiveStageMode != "closed_switch_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_pre_writer_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source switch state/action not carried")
	}
	if report.SourceSwitchKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory causal prefix mismatch")
	}
	if !strings.HasPrefix(report.WriterInventoryHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-read-") ||
		report.WriterInventoryHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source writer preflight mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID, "weighted-resonance-graft-admission-final-gate-observation-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-receiver-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory causal_id mismatch")
	}
	if report.WriterInventoryHash == "" || report.WriterInventoryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory writer_inventory_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory read_back_hash mismatch")
	}
	if report.WriterInventoryHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport) string {
	h := hashJSON(struct {
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		SourceLiveStageID         string `json:"source_live_stage_id"`
		Target                    string `json:"target"`
		WriterInventoryKind       string `json:"writer_inventory_kind"`
		WriterInventoryStage      string `json:"writer_inventory_stage"`
	}{
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
		SourceLiveStageID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		Target:                    sw.Target,
		WriterInventoryKind:       sw.WriterInventoryKind,
		WriterInventoryStage:      sw.WriterInventoryStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport) string {
	h := hashJSON(struct {
		CausalID                  string `json:"causal_id"`
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightHash string `json:"source_writer_preflight_hash"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		WriterState               string `json:"writer_state"`
		WriterAction              string `json:"writer_action"`
		RollbackState             string `json:"rollback_state"`
		RollbackAction            string `json:"rollback_action"`
		InventoryState            string `json:"inventory_state"`
		InventoryAction           string `json:"inventory_action"`
		WriterContract            string `json:"writer_contract"`
		RollbackContract          string `json:"rollback_contract"`
		AdmissionLedgerContract   string `json:"admission_ledger_contract"`
		ContractsReady            bool   `json:"contracts_ready"`
		StageState                string `json:"stage_state"`
		StageAction               string `json:"stage_action"`
		EnableState               string `json:"enable_state"`
		EnableAction              string `json:"enable_action"`
		SwitchState               string `json:"switch_state"`
		SwitchAction              string `json:"switch_action"`
		Promotion                 string `json:"promotion"`
		Action                    string `json:"action"`
		ReceiptShape              string `json:"receipt_shape"`
		WriterInventoryMode       string `json:"writer_inventory_mode"`
		WriterPreflightVerified   bool   `json:"writer_preflight_verified"`
		LiveStageVerified         bool   `json:"live_stage_verified"`
		RequiresWriter            bool   `json:"requires_writer"`
		WriterReady               bool   `json:"writer_ready"`
		RequiresRollback          bool   `json:"requires_rollback"`
		RollbackReady             bool   `json:"rollback_ready"`
		ReadOnly                  bool   `json:"read_only"`
		ReplayOnly                bool   `json:"replay_only"`
		AdmissionRequired         bool   `json:"admission_required"`
		ShadowOnly                bool   `json:"shadow_only"`
		DryRunOnly                bool   `json:"dry_run_only"`
		GraftAllowed              bool   `json:"graft_allowed"`
		BodyMutation              bool   `json:"body_mutation_allowed"`
		LiveAdmission             bool   `json:"live_admission_enabled"`
	}{
		CausalID:                  sw.CausalID,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
		WriterState:               sw.WriterState,
		WriterAction:              sw.WriterAction,
		RollbackState:             sw.RollbackState,
		RollbackAction:            sw.RollbackAction,
		InventoryState:            sw.InventoryState,
		InventoryAction:           sw.InventoryAction,
		WriterContract:            sw.WriterContract,
		RollbackContract:          sw.RollbackContract,
		AdmissionLedgerContract:   sw.AdmissionLedgerContract,
		ContractsReady:            sw.ContractsReady,
		StageState:                sw.StageState,
		StageAction:               sw.StageAction,
		EnableState:               sw.EnableState,
		EnableAction:              sw.EnableAction,
		SwitchState:               sw.SwitchState,
		SwitchAction:              sw.SwitchAction,
		Promotion:                 sw.Promotion,
		Action:                    sw.Action,
		ReceiptShape:              sw.ReceiptShape,
		WriterInventoryMode:       sw.WriterInventoryMode,
		WriterPreflightVerified:   sw.WriterPreflightVerified,
		LiveStageVerified:         sw.LiveStageVerified,
		RequiresWriter:            sw.RequiresWriter,
		WriterReady:               sw.WriterReady,
		RequiresRollback:          sw.RequiresRollback,
		RollbackReady:             sw.RollbackReady,
		ReadOnly:                  sw.ReadOnly,
		ReplayOnly:                sw.ReplayOnly,
		AdmissionRequired:         sw.AdmissionRequired,
		ShadowOnly:                sw.ShadowOnly,
		DryRunOnly:                sw.DryRunOnly,
		GraftAllowed:              sw.GraftAllowed,
		BodyMutation:              sw.BodyMutationAllowed,
		LiveAdmission:             sw.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport) string {
	h := hashJSON(struct {
		WriterInventoryHash       string `json:"writer_inventory_hash"`
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		WriterInventoryKind       string `json:"writer_inventory_kind"`
		WriterInventoryReady      bool   `json:"writer_inventory_ready"`
		WriterPreflightConsumed   bool   `json:"writer_preflight_consumed"`
		RequiresWriter            bool   `json:"requires_writer"`
		WriterReady               bool   `json:"writer_ready"`
		RequiresRollback          bool   `json:"requires_rollback"`
		RollbackReady             bool   `json:"rollback_ready"`
		ContractsReady            bool   `json:"contracts_ready"`
		LiveReady                 bool   `json:"live_ready"`
		BodyMutation              bool   `json:"body_mutation"`
		LiveAdmission             bool   `json:"live_admission"`
		WriteAllowed              bool   `json:"write_allowed"`
		AdmissionAllowed          bool   `json:"admission_allowed"`
	}{
		WriterInventoryHash:       sw.WriterInventoryHash,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
		WriterInventoryKind:       sw.WriterInventoryKind,
		WriterInventoryReady:      sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady,
		WriterPreflightConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightConsumed,
		RequiresWriter:            sw.RequiresWriter,
		WriterReady:               sw.WriterReady,
		RequiresRollback:          sw.RequiresRollback,
		RollbackReady:             sw.RollbackReady,
		ContractsReady:            sw.ContractsReady,
		LiveReady:                 sw.LiveReady,
		BodyMutation:              sw.BodyMutationAllowed,
		LiveAdmission:             sw.LiveAdmissionEnabled,
		WriteAllowed:              sw.WriteAllowed,
		AdmissionAllowed:          sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport) string {
	h := hashJSON(struct {
		Schema                    string `json:"schema"`
		Status                    string `json:"status"`
		Action                    string `json:"action"`
		WriterState               string `json:"writer_state"`
		WriterAction              string `json:"writer_action"`
		RollbackState             string `json:"rollback_state"`
		RollbackAction            string `json:"rollback_action"`
		InventoryState            string `json:"inventory_state"`
		InventoryAction           string `json:"inventory_action"`
		WriterContract            string `json:"writer_contract"`
		RollbackContract          string `json:"rollback_contract"`
		AdmissionLedgerContract   string `json:"admission_ledger_contract"`
		StageState                string `json:"stage_state"`
		StageAction               string `json:"stage_action"`
		EnableState               string `json:"enable_state"`
		EnableAction              string `json:"enable_action"`
		SwitchState               string `json:"switch_state"`
		SwitchAction              string `json:"switch_action"`
		Promotion                 string `json:"promotion"`
		SourceReport              string `json:"source_report"`
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightHash string `json:"source_writer_preflight_hash"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		SourceLiveStageID         string `json:"source_live_stage_id"`
		SourceLiveStageHash       string `json:"source_live_stage_hash"`
		SourceLiveStageRead       string `json:"source_live_stage_read_back_hash"`
		SourceEnableGateID        string `json:"source_enable_gate_id"`
		SourceEnableGateHash      string `json:"source_enable_gate_hash"`
		SourceEnableGateRead      string `json:"source_enable_gate_read_back_hash"`
		SourceSwitchID            string `json:"source_switch_id"`
		SourceSwitchHash          string `json:"source_switch_hash"`
		SourceSwitchRead          string `json:"source_switch_read_back_hash"`
		SourcePromotionID         string `json:"source_promotion_id"`
		SourceDecisionID          string `json:"source_decision_id"`
		SourceProofID             string `json:"source_proof_id"`
		SourceReaderID            string `json:"source_reader_id"`
		SourceStoreID             string `json:"source_store_id"`
		SourceCandidateID         string `json:"source_candidate_id"`
		SourceGateID              string `json:"source_gate_id"`
		SourcePreflightID         string `json:"source_preflight_id"`
		SourceBoundaryID          string `json:"source_boundary_id"`
		SourceObservationID       string `json:"source_observation_id"`
		SourceReceiverID          string `json:"source_receiver_id"`
		CausalID                  string `json:"causal_id"`
		WriterInventoryHash       string `json:"writer_inventory_hash"`
		ReadBackHash              string `json:"read_back_hash"`
		Ready                     bool   `json:"ready"`
		ReceiptShape              string `json:"receipt_shape"`
		WriterInventoryKind       string `json:"writer_inventory_kind"`
		WriterInventoryMode       string `json:"writer_inventory_mode"`
		WriterInventoryStage      string `json:"writer_inventory_stage"`
		WriterPreflightVerified   bool   `json:"writer_preflight_verified"`
		LiveStageVerified         bool   `json:"live_stage_verified"`
		EnableGateVerified        bool   `json:"enable_gate_verified"`
		SwitchVerified            bool   `json:"switch_verified"`
		PromotionVerified         bool   `json:"promotion_verified"`
		AdmissionRequired         bool   `json:"admission_required"`
		ShadowOnly                bool   `json:"shadow_only"`
		GraftAllowed              bool   `json:"graft_allowed"`
		DryRunOnly                bool   `json:"dry_run_only"`
		RawDreamTextAllowed       bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed       bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed       bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed       bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed       bool   `json:"body_mutation_allowed"`
		RequiresWriter            bool   `json:"requires_writer"`
		WriterReady               bool   `json:"writer_ready"`
		RollbackRequired          bool   `json:"rollback_required"`
		RequiresRollback          bool   `json:"requires_rollback"`
		RollbackReady             bool   `json:"rollback_ready"`
		ReadOnly                  bool   `json:"read_only"`
		ReplayOnly                bool   `json:"replay_only"`
		LiveReady                 bool   `json:"live_ready"`
		ContractsReady            bool   `json:"contracts_ready"`
		BodyTarget                string `json:"body_target"`
		WriteAllowed              bool   `json:"write_allowed"`
		AdmissionAllowed          bool   `json:"admission_allowed"`
		LiveAdmissionEnabled      bool   `json:"live_admission_enabled"`
		MutatesState              bool   `json:"mutates_state"`
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory"`
		SourcePromotionReady      bool   `json:"source_promotion_ready"`
		SourceDecisionReady       bool   `json:"source_decision_ready"`
		SourcePreconditionReady   bool   `json:"source_precondition_ready"`
		SourceProofReady          bool   `json:"source_proof_ready"`
		SourceReaderReady         bool   `json:"source_reader_ready"`
		SourceStoreReady          bool   `json:"source_store_ready"`
		SourceCandidateReady      bool   `json:"source_candidate_ready"`
		SourceGateReady           bool   `json:"source_gate_ready"`
		SourcePreflightReady      bool   `json:"source_preflight_ready"`
		SourceBoundaryReady       bool   `json:"source_boundary_ready"`
		SourceObservationReady    bool   `json:"source_observation_ready"`
		SourceReceiverReady       bool   `json:"source_receiver_ready"`
		SourceIntentReady         bool   `json:"source_intent_ready"`
		SourceFinalGateReady      bool   `json:"source_final_gate_ready"`
		SourceSealReady           bool   `json:"source_seal_ready"`
		SourcePermitReady         bool   `json:"source_permit_ready"`
		SourceAuthorityUsed       bool   `json:"source_authority_consumed"`
		SourceAuthorityNeeded     bool   `json:"source_authority_required"`
	}{
		Schema:                    sw.Schema,
		Status:                    sw.Status,
		Action:                    sw.Action,
		WriterState:               sw.WriterState,
		WriterAction:              sw.WriterAction,
		RollbackState:             sw.RollbackState,
		RollbackAction:            sw.RollbackAction,
		InventoryState:            sw.InventoryState,
		InventoryAction:           sw.InventoryAction,
		WriterContract:            sw.WriterContract,
		RollbackContract:          sw.RollbackContract,
		AdmissionLedgerContract:   sw.AdmissionLedgerContract,
		StageState:                sw.StageState,
		StageAction:               sw.StageAction,
		EnableState:               sw.EnableState,
		EnableAction:              sw.EnableAction,
		SwitchState:               sw.SwitchState,
		SwitchAction:              sw.SwitchAction,
		Promotion:                 sw.Promotion,
		SourceReport:              sw.SourceReport,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
		SourceLiveStageID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		SourceLiveStageHash:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash,
		SourceLiveStageRead:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack,
		SourceEnableGateID:        sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		SourceEnableGateHash:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash,
		SourceEnableGateRead:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack,
		SourceSwitchID:            sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		SourceSwitchHash:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash,
		SourceSwitchRead:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack,
		SourcePromotionID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceDecisionID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceProofID:             sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceReaderID:            sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:             sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceGateID:              sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourcePreflightID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceBoundaryID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceObservationID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceReceiverID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		CausalID:                  sw.CausalID,
		WriterInventoryHash:       sw.WriterInventoryHash,
		ReadBackHash:              sw.ReadBackHash,
		Ready:                     sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady,
		ReceiptShape:              sw.ReceiptShape,
		WriterInventoryKind:       sw.WriterInventoryKind,
		WriterInventoryMode:       sw.WriterInventoryMode,
		WriterInventoryStage:      sw.WriterInventoryStage,
		WriterPreflightVerified:   sw.WriterPreflightVerified,
		LiveStageVerified:         sw.LiveStageVerified,
		EnableGateVerified:        sw.EnableGateVerified,
		SwitchVerified:            sw.SwitchVerified,
		PromotionVerified:         sw.PromotionVerified,
		AdmissionRequired:         sw.AdmissionRequired,
		ShadowOnly:                sw.ShadowOnly,
		GraftAllowed:              sw.GraftAllowed,
		DryRunOnly:                sw.DryRunOnly,
		RawDreamTextAllowed:       sw.RawDreamTextAllowed,
		JanusSurfaceAllowed:       sw.JanusSurfaceAllowed,
		CoocLearningAllowed:       sw.CoocLearningAllowed,
		DeltaHarvestAllowed:       sw.DeltaHarvestAllowed,
		BodyMutationAllowed:       sw.BodyMutationAllowed,
		RequiresWriter:            sw.RequiresWriter,
		WriterReady:               sw.WriterReady,
		RollbackRequired:          sw.RollbackRequired,
		RequiresRollback:          sw.RequiresRollback,
		RollbackReady:             sw.RollbackReady,
		ReadOnly:                  sw.ReadOnly,
		ReplayOnly:                sw.ReplayOnly,
		LiveReady:                 sw.LiveReady,
		ContractsReady:            sw.ContractsReady,
		BodyTarget:                sw.BodyTarget,
		WriteAllowed:              sw.WriteAllowed,
		AdmissionAllowed:          sw.AdmissionAllowed,
		LiveAdmissionEnabled:      sw.LiveAdmissionEnabled,
		MutatesState:              sw.MutatesState,
		NextStepBlockedWithout:    sw.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventory,
		SourcePromotionReady:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourceDecisionReady:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourcePreconditionReady:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceProofReady:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceReaderReady:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceStoreReady:          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceCandidateReady:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceGateReady:           sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourcePreflightReady:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceBoundaryReady:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceObservationReady:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceReceiverReady:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceIntentReady:         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceFinalGateReady:      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceSealReady:           sw.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourcePermitReady:         sw.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceAuthorityUsed:       sw.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:     sw.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory decode failed: %w", err)
	}
	return report, root, nil
}
