package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport struct {
	Schema                                                                                                                                                                                                                               string `json:"schema"`
	Status                                                                                                                                                                                                                               string `json:"status"`
	Target                                                                                                                                                                                                                               string `json:"target"`
	TargetKind                                                                                                                                                                                                                           string `json:"target_kind"`
	TargetMode                                                                                                                                                                                                                           string `json:"target_mode"`
	Action                                                                                                                                                                                                                               string `json:"action"`
	LedgerState                                                                                                                                                                                                                          string `json:"ledger_state"`
	LedgerAction                                                                                                                                                                                                                         string `json:"ledger_action"`
	LedgerContract                                                                                                                                                                                                                       string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                                                                                                     string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                                                                                                   string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                                                                                                     string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                                                                                          bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                                                                                                  bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateConsumed   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateRequired   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore                                                                                                                                 bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id"`
	ReceiptShape                                                                                                                                                                                                                         string `json:"receipt_shape"`
	StoreKind                                                                                                                                                                                                                            string `json:"store_kind"`
	StoreMode                                                                                                                                                                                                                            string `json:"store_mode"`
	StoreStage                                                                                                                                                                                                                           string `json:"store_stage"`
	CausalID                                                                                                                                                                                                                             string `json:"causal_id"`
	StoreHash                                                                                                                                                                                                                            string `json:"store_hash"`
	ReadBackHash                                                                                                                                                                                                                         string `json:"read_back_hash"`
	CandidateVerified                                                                                                                                                                                                                    bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                                                                                         bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                                                                                                    bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                                                                                                     bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                                                                                                  bool   `json:"observation_verified"`
	FinalGateVerified                                                                                                                                                                                                                    bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                                                                                         bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                                                                                                       bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                                                                                                    bool   `json:"authority_verified"`
	AdmissionRequired                                                                                                                                                                                                                    bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                                                                                           bool   `json:"shadow_only"`
	DryRunOnly                                                                                                                                                                                                                           bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                                                                                            bool   `json:"live_ready"`
	RollbackRequired                                                                                                                                                                                                                     bool   `json:"rollback_required"`
	AppendOnly                                                                                                                                                                                                                           bool   `json:"append_only"`
	ReadBack                                                                                                                                                                                                                             bool   `json:"read_back"`
	ReceiptPersisted                                                                                                                                                                                                                     bool   `json:"receipt_persisted"`
	ReceiptVerified                                                                                                                                                                                                                      bool   `json:"receipt_verified"`
	RawDreamTextAllowed                                                                                                                                                                                                                  bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                                                                                                 bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                                                                                                bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                                                                                                  bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                                                                                                  bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                                                                                                  bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                                                                                                  bool   `json:"body_mutation_allowed"`
	AuthorityGranted                                                                                                                                                                                                                     bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                                                                                                       bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                                                                                         bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                                                                                                     bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                                                                                                 bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                                                                                         bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                                                                                           string `json:"body_target"`
	Passed                                                                                                                                                                                                                               bool   `json:"passed"`
	Reason                                                                                                                                                                                                                               string `json:"reason"`

	SourceSchema                                                                                                                                                                                                                             string `json:"source_schema"`
	SourceStatus                                                                                                                                                                                                                             string `json:"source_status"`
	SourceTarget                                                                                                                                                                                                                             string `json:"source_target"`
	SourceReport                                                                                                                                                                                                                             string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason                                                                                                                                                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_reason"`
	SourceCandidateReceiptShape                                                                                                                                                                                                              string `json:"source_candidate_receipt_shape"`
	SourceCandidateState                                                                                                                                                                                                                     string `json:"source_candidate_state"`
	SourceCandidateAction                                                                                                                                                                                                                    string `json:"source_candidate_action"`
	SourceCandidateTarget                                                                                                                                                                                                                    string `json:"source_candidate_target"`
	SourceCandidateTargetKind                                                                                                                                                                                                                string `json:"source_candidate_target_kind"`
	SourceCandidateTargetMode                                                                                                                                                                                                                string `json:"source_candidate_target_mode"`
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
	SourceCandidateJanusSurfaceAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_janus_surface_allowed"`
	SourceCandidateCoocLearningAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_cooc_learning_allowed"`
	SourceCandidateDeltaHarvestAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_delta_harvest_allowed"`
	SourceCandidateBodyMutationAllowed                                                                                                                                                                                                       bool   `json:"source_candidate_body_mutation_allowed"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateHash                                                                                                                                                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash                                                                                                                                                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                                                                                                   bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly                                                                                                                                                              bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified                                                                                                                                                       bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified                                                                                                                                                     bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified                                                                                                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved                                                                                                                                                             bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded                                                                                                                                                            bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_janus_surface_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_cooc_learning_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed                                                                                                                                                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_delta_harvest_allowed"`
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
	SourceLedgerReady                                                                                                                                                                                                        bool   `json:"source_ledger_ready"`
	SourceLedgerAppendAllowed                                                                                                                                                                                                bool   `json:"source_ledger_append_allowed"`
	SourceAuthorityGranted                                                                                                                                                                                                   bool   `json:"source_authority_granted"`
	SourceContractsReady                                                                                                                                                                                                     bool   `json:"source_contracts_ready"`
	SourceWriteAllowed                                                                                                                                                                                                       bool   `json:"source_write_allowed"`
	SourceAdmissionAllowed                                                                                                                                                                                                   bool   `json:"source_admission_allowed"`
	SourceLiveAdmissionEnabled                                                                                                                                                                                               bool   `json:"source_live_admission_enabled"`
	SourceMutatesState                                                                                                                                                                                                       bool   `json:"source_mutates_state"`
	SourceBodyMutationAllowed                                                                                                                                                                                                bool   `json:"source_body_mutation_allowed"`
	SourceBodyTarget                                                                                                                                                                                                         string `json:"source_body_target"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStore(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT")
	}
	candidatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store output path missing")
	}
	candidate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportForAssert(candidatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReportError(candidate, root); err != nil {
		return err
	}
	store := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store",
		TargetMode:          "append_only_read_back_store_dry_run",
		Action:              "store_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run",
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateConsumed:   true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateRequired:   true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore:                                                                                                                                 true,
		ReceiptShape:          "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt",
		StoreKind:             "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store",
		StoreMode:             "append_only_read_back_store",
		StoreStage:            "post_preflight_gate_candidate_pre_live_admission_store",
		CandidateVerified:     true,
		GateVerified:          candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified,
		PreflightVerified:     candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified,
		BoundaryVerified:      candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified,
		ObservationVerified:   candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified,
		FinalGateVerified:     candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SealVerified:          candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		PermitVerified:        candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady,
		AuthorityVerified:     candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		AdmissionRequired:     true,
		ShadowOnly:            true,
		DryRunOnly:            true,
		LiveReady:             candidate.LiveReady,
		RollbackRequired:      true,
		AppendOnly:            true,
		ReadBack:              true,
		ReceiptPersisted:      true,
		ReceiptVerified:       true,
		RawDreamTextAllowed:   false,
		RawDreamTextObserved:  false,
		RawDreamTextForwarded: false,
		JanusSurfaceAllowed:   false,
		CoocLearningAllowed:   false,
		DeltaHarvestAllowed:   false,
		BodyMutationAllowed:   false,
		AuthorityGranted:      false,
		ContractsReady:        false,
		WriteAllowed:          false,
		AdmissionAllowed:      false,
		LiveAdmissionEnabled:  false,
		MutatesState:          false,
		BodyTarget:            "none",
		Passed:                true,
		Reason:                "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate stored without ledger append or body mutation",

		SourceSchema: candidate.Schema,
		SourceStatus: candidate.Status,
		SourceTarget: candidate.Target,
		SourceReport: candidatePath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID:       candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady:    candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID: candidate.CausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash:         candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash: candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason:       candidate.Reason,
		SourceCandidateReceiptShape:          candidate.ReceiptShape,
		SourceCandidateState:                 candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateState,
		SourceCandidateAction:                candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction,
		SourceCandidateTarget:                candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTarget,
		SourceCandidateTargetKind:            candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetKind,
		SourceCandidateTargetMode:            candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetMode,
		SourceCandidateKind:                  candidate.FinalGateObservationBoundaryPreflightGateCandidateKind,
		SourceCandidateMode:                  candidate.FinalGateObservationBoundaryPreflightGateCandidateMode,
		SourceCandidateStage:                 candidate.FinalGateObservationBoundaryPreflightGateCandidateStage,
		SourceCandidateDryRunOnly:            candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly,
		SourceCandidateGateVerified:          candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified,
		SourceCandidatePreflightVerified:     candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified,
		SourceCandidateBoundaryVerified:      candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified,
		SourceCandidateObservationVerified:   candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified,
		SourceCandidateReadBackVerified:      candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified,
		SourceCandidateOpened:                candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceCandidateRawDreamTextObserved:  candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved,
		SourceCandidateRawDreamTextForwarded: candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded,
		SourceCandidateRawDreamTextAllowed:   candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextAllowed,
		SourceCandidateJanusSurfaceAllowed:   candidate.FinalGateObservationBoundaryPreflightGateCandidateJanusSurfaceAllowed,
		SourceCandidateCoocLearningAllowed:   candidate.FinalGateObservationBoundaryPreflightGateCandidateCoocLearningAllowed,
		SourceCandidateDeltaHarvestAllowed:   candidate.FinalGateObservationBoundaryPreflightGateCandidateDeltaHarvestAllowed,
		SourceCandidateBodyMutationAllowed:   candidate.FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID:       candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady:    candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID: candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausal,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:                candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash:        candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:               candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:          candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified:   candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified:    candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified: candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified:    candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved:         candidate.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:        candidate.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:          candidate.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed:          candidate.SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed,
		SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed:          candidate.SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed,
		SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed:          candidate.SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed,
		SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed:          candidate.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID:    candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady: candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID:             candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady:          candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID:                     candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady:                  candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID:                                candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady:                             candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady:                               candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady:                                     candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady:                                          candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady:                                     candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady:                                        candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStagePermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady:                                     candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadinessReady,
		SourceWriterInventoryVerified: candidate.WriterInventoryVerified,
		SourceWriterPreflightVerified: candidate.WriterPreflightVerified,
		SourceAdmissionRequired:       candidate.AdmissionRequired,
		SourceShadowOnly:              candidate.ShadowOnly,
		SourceDryRunOnly:              candidate.DryRunOnly,
		SourceRequiresWriter:          candidate.RequiresWriter,
		SourceRollbackRequired:        candidate.RollbackRequired,
		SourceRequiresRollback:        candidate.RequiresRollback,
		SourceReadOnly:                candidate.ReadOnly,
		SourceReplayOnly:              candidate.ReplayOnly,
		SourceLedgerReady:             candidate.LedgerReady,
		SourceLedgerAppendAllowed:     candidate.LedgerAppendAllowed,
		SourceAuthorityGranted:        candidate.AuthorityGranted,
		SourceContractsReady:          candidate.ContractsReady,
		SourceWriteAllowed:            candidate.WriteAllowed,
		SourceAdmissionAllowed:        candidate.AdmissionAllowed,
		SourceLiveAdmissionEnabled:    candidate.LiveAdmissionEnabled,
		SourceMutatesState:            candidate.MutatesState,
		SourceBodyMutationAllowed:     candidate.BodyMutationAllowed,
		SourceBodyTarget:              candidate.BodyTarget,
	}
	store.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID(store)
	store.StoreHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreHash(store)
	store.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReadBackHash(store)
	store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID(store)
	if store.CausalID == "" ||
		store.StoreHash == "" ||
		store.ReadBackHash == "" ||
		store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID == "" ||
		store.StoreHash == store.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store read-back proof failed")
	}
	raw, err := json.MarshalIndent(store, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_report=%s\n", outputPath, candidatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run")
	}
	if report.Target != "live_route_admission_next_step" ||
		report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		report.TargetMode != "append_only_read_back_store_dry_run" ||
		report.Action != "store_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store route shape mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ledger_append" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt" ||
		report.StoreKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		report.StoreMode != "append_only_read_back_store" ||
		report.StoreStage != "post_preflight_gate_candidate_pre_live_admission_store" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore},
		{"candidate_verified", report.CandidateVerified},
		{"gate_verified", report.GateVerified},
		{"preflight_verified", report.PreflightVerified},
		{"boundary_verified", report.BoundaryVerified},
		{"observation_verified", report.ObservationVerified},
		{"final_gate_verified", report.FinalGateVerified},
		{"seal_verified", report.SealVerified},
		{"permit_verified", report.PermitVerified},
		{"authority_verified", report.AuthorityVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"append_only", report.AppendOnly},
		{"read_back", report.ReadBack},
		{"receipt_persisted", report.ReceiptPersisted},
		{"receipt_verified", report.ReceiptVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateReady},
		{"source_candidate_dry_run_only", report.SourceCandidateDryRunOnly},
		{"source_candidate_gate_verified", report.SourceCandidateGateVerified},
		{"source_candidate_preflight_verified", report.SourceCandidatePreflightVerified},
		{"source_candidate_boundary_verified", report.SourceCandidateBoundaryVerified},
		{"source_candidate_observation_verified", report.SourceCandidateObservationVerified},
		{"source_candidate_read_back_verified", report.SourceCandidateReadBackVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady},
		{"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
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
		{"source_candidate_opened", report.SourceCandidateOpened},
		{"source_candidate_raw_dream_text_observed", report.SourceCandidateRawDreamTextObserved},
		{"source_candidate_raw_dream_text_forwarded", report.SourceCandidateRawDreamTextForwarded},
		{"source_candidate_raw_dream_text_allowed", report.SourceCandidateRawDreamTextAllowed},
		{"source_candidate_janus_surface_allowed", report.SourceCandidateJanusSurfaceAllowed},
		{"source_candidate_cooc_learning_allowed", report.SourceCandidateCoocLearningAllowed},
		{"source_candidate_delta_harvest_allowed", report.SourceCandidateDeltaHarvestAllowed},
		{"source_candidate_body_mutation_allowed", report.SourceCandidateBodyMutationAllowed},
		{"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded},
		{"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed", report.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_janus_surface_allowed", report.SourceFinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_cooc_learning_allowed", report.SourceFinalGateObservationBoundaryPreflightGateCoocLearningAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_delta_harvest_allowed", report.SourceFinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed},
		{"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed", report.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed},
		{"source_ledger_ready", report.SourceLedgerReady},
		{"source_ledger_append_allowed", report.SourceLedgerAppendAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"source_contracts_ready", report.SourceContractsReady},
		{"source_write_allowed", report.SourceWriteAllowed},
		{"source_admission_allowed", report.SourceAdmissionAllowed},
		{"source_live_admission_enabled", report.SourceLiveAdmissionEnabled},
		{"source_mutates_state", report.SourceMutatesState},
		{"source_body_mutation_allowed", report.SourceBodyMutationAllowed},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_store_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID},
		{"causal_id", report.CausalID},
		{"store_hash", report.StoreHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_reason", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_blocked_dry_run" ||
		report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source route mismatch")
	}
	if report.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate_candidate_receipt" ||
		report.SourceCandidateState != "blocked" ||
		report.SourceCandidateAction != "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.SourceCandidateTarget != "resonance" ||
		report.SourceCandidateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_final_gate_receiver_observation_boundary_preflight_gate" ||
		report.SourceCandidateTargetMode != "closed_preflight_gate_candidate_dry_run" ||
		report.SourceCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.SourceCandidateMode != "no_mutation_preflight_gate_candidate" ||
		report.SourceCandidateStage != "post_preflight_gate_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source candidate shape mismatch")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source candidate reason mismatch: got %q", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason)
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-id-") ||
		!strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-causal-") ||
		!strings.HasPrefix(report.StoreHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-") ||
		!strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-read-") ||
		report.StoreHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store proof prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source chain prefix mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.SourceBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store source_body_target mismatch: got %q want %q", report.SourceBodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store causal_id mismatch")
	}
	if report.StoreHash == "" || report.StoreHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store store_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store read_back_hash mismatch")
	}
	if report.StoreHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate stored without ledger append or body mutation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreCausalID(store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport) string {
	h := hashJSON(struct {
		SourceCandidateID           string `json:"source_candidate_id"`
		SourceCandidateReadBackHash string `json:"source_candidate_read_back_hash"`
		SourceGateID                string `json:"source_gate_id"`
		SourcePreflightID           string `json:"source_preflight_id"`
		SourceBoundaryID            string `json:"source_boundary_id"`
		SourceObservationID         string `json:"source_observation_id"`
		SourceReceiverID            string `json:"source_receiver_id"`
		Target                      string `json:"target"`
		StoreKind                   string `json:"store_kind"`
		StoreStage                  string `json:"store_stage"`
	}{
		SourceCandidateID:           store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceCandidateReadBackHash: store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		SourceGateID:                store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourcePreflightID:           store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceBoundaryID:            store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceObservationID:         store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceReceiverID:            store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		Target:                      store.Target,
		StoreKind:                   store.StoreKind,
		StoreStage:                  store.StoreStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreHash(store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport) string {
	h := hashJSON(struct {
		CausalID                    string `json:"causal_id"`
		SourceCandidateID           string `json:"source_candidate_id"`
		SourceCandidateHash         string `json:"source_candidate_hash"`
		SourceCandidateReadBackHash string `json:"source_candidate_read_back_hash"`
		StoreMode                   string `json:"store_mode"`
		AppendOnly                  bool   `json:"append_only"`
		ReadBack                    bool   `json:"read_back"`
		ReceiptPersisted            bool   `json:"receipt_persisted"`
		ReceiptVerified             bool   `json:"receipt_verified"`
		CandidateVerified           bool   `json:"candidate_verified"`
		DryRunOnly                  bool   `json:"dry_run_only"`
		AdmissionRequired           bool   `json:"admission_required"`
		SourceCandidateOpened       bool   `json:"source_candidate_opened"`
		LedgerAppendAllowed         bool   `json:"ledger_append_allowed"`
		BodyMutationAllowed         bool   `json:"body_mutation_allowed"`
	}{
		CausalID:                    store.CausalID,
		SourceCandidateID:           store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceCandidateHash:         store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceCandidateReadBackHash: store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		StoreMode:                   store.StoreMode,
		AppendOnly:                  store.AppendOnly,
		ReadBack:                    store.ReadBack,
		ReceiptPersisted:            store.ReceiptPersisted,
		ReceiptVerified:             store.ReceiptVerified,
		CandidateVerified:           store.CandidateVerified,
		DryRunOnly:                  store.DryRunOnly,
		AdmissionRequired:           store.AdmissionRequired,
		SourceCandidateOpened:       store.SourceCandidateOpened,
		LedgerAppendAllowed:         store.LedgerAppendAllowed,
		BodyMutationAllowed:         store.BodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReadBackHash(store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport) string {
	h := hashJSON(struct {
		StoreHash       string `json:"store_hash"`
		SourceCandidate string `json:"source_candidate_id"`
		StoreKind       string `json:"store_kind"`
		StoreReady      bool   `json:"store_ready"`
		ReceiptVerified bool   `json:"receipt_verified"`
		BodyMutation    bool   `json:"body_mutation"`
		AdmissionOpen   bool   `json:"admission_open"`
	}{
		StoreHash:       store.StoreHash,
		SourceCandidate: store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		StoreKind:       store.StoreKind,
		StoreReady:      store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady,
		ReceiptVerified: store.ReceiptVerified,
		BodyMutation:    store.BodyMutationAllowed,
		AdmissionOpen:   store.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreID(store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		SourcePreflightID      string `json:"source_preflight_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		SourceReceiverID       string `json:"source_receiver_id"`
		CausalID               string `json:"causal_id"`
		StoreHash              string `json:"store_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReceiptShape           string `json:"receipt_shape"`
		StoreKind              string `json:"store_kind"`
		StoreMode              string `json:"store_mode"`
		StoreStage             string `json:"store_stage"`
		AppendOnly             bool   `json:"append_only"`
		ReadBack               bool   `json:"read_back"`
		ReceiptPersisted       bool   `json:"receipt_persisted"`
		ReceiptVerified        bool   `json:"receipt_verified"`
		CandidateVerified      bool   `json:"candidate_verified"`
		GateVerified           bool   `json:"gate_verified"`
		PreflightVerified      bool   `json:"preflight_verified"`
		BoundaryVerified       bool   `json:"boundary_verified"`
		ObservationVerified    bool   `json:"observation_verified"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		DryRunOnly             bool   `json:"dry_run_only"`
		LiveReady              bool   `json:"live_ready"`
		LedgerAppendAllowed    bool   `json:"ledger_append_allowed"`
		BodyTarget             string `json:"body_target"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store"`
		SourceCandidateOpened  bool   `json:"source_candidate_opened"`
		SourceGateReady        bool   `json:"source_gate_ready"`
	}{
		Schema:                 store.Schema,
		Status:                 store.Status,
		Action:                 store.Action,
		SourceReport:           store.SourceReport,
		SourceCandidateID:      store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateID,
		SourceGateID:           store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateID,
		SourcePreflightID:      store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightID,
		SourceBoundaryID:       store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryID,
		SourceObservationID:    store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationID,
		SourceReceiverID:       store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverID,
		CausalID:               store.CausalID,
		StoreHash:              store.StoreHash,
		ReadBackHash:           store.ReadBackHash,
		Ready:                  store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReady,
		ReceiptShape:           store.ReceiptShape,
		StoreKind:              store.StoreKind,
		StoreMode:              store.StoreMode,
		StoreStage:             store.StoreStage,
		AppendOnly:             store.AppendOnly,
		ReadBack:               store.ReadBack,
		ReceiptPersisted:       store.ReceiptPersisted,
		ReceiptVerified:        store.ReceiptVerified,
		CandidateVerified:      store.CandidateVerified,
		GateVerified:           store.GateVerified,
		PreflightVerified:      store.PreflightVerified,
		BoundaryVerified:       store.BoundaryVerified,
		ObservationVerified:    store.ObservationVerified,
		AdmissionRequired:      store.AdmissionRequired,
		ShadowOnly:             store.ShadowOnly,
		DryRunOnly:             store.DryRunOnly,
		LiveReady:              store.LiveReady,
		LedgerAppendAllowed:    store.LedgerAppendAllowed,
		BodyTarget:             store.BodyTarget,
		WriteAllowed:           store.WriteAllowed,
		AdmissionAllowed:       store.AdmissionAllowed,
		LiveAdmissionEnabled:   store.LiveAdmissionEnabled,
		MutatesState:           store.MutatesState,
		NextStepBlockedWithout: store.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStore,
		SourceCandidateOpened:  store.SourceCandidateOpened,
		SourceGateReady:        store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-final-gate-receiver-observation-boundary-preflight-gate-candidate-store-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageFinalGateReceiverObservationBoundaryPreflightGateCandidateStoreReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage final gate receiver observation boundary preflight gate candidate store decode failed: %w", err)
	}
	return report, root, nil
}
