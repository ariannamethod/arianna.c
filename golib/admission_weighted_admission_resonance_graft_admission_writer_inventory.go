package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema = "arianna.live_route_weighted_admission_resonance_graft_admission_writer_inventory.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport struct {
	Schema                                                                string `json:"schema"`
	Status                                                                string `json:"status"`
	Target                                                                string `json:"target"`
	TargetKind                                                            string `json:"target_kind"`
	TargetMode                                                            string `json:"target_mode"`
	Action                                                                string `json:"action"`
	WriterState                                                           string `json:"writer_state"`
	WriterAction                                                          string `json:"writer_action"`
	RollbackState                                                         string `json:"rollback_state"`
	RollbackAction                                                        string `json:"rollback_action"`
	StageState                                                            string `json:"stage_state"`
	StageAction                                                           string `json:"stage_action"`
	EnableState                                                           string `json:"enable_state"`
	EnableAction                                                          string `json:"enable_action"`
	SwitchState                                                           string `json:"switch_state"`
	SwitchAction                                                          string `json:"switch_action"`
	Promotion                                                             string `json:"promotion"`
	InventoryState                                                        string `json:"inventory_state"`
	InventoryAction                                                       string `json:"inventory_action"`
	WriterContract                                                        string `json:"writer_contract"`
	RollbackContract                                                      string `json:"rollback_contract"`
	AdmissionLedgerContract                                               string `json:"admission_ledger_contract"`
	WriterContractPresent                                                 bool   `json:"writer_contract_present"`
	RollbackContractPresent                                               bool   `json:"rollback_contract_present"`
	LedgerContractPresent                                                 bool   `json:"ledger_contract_present"`
	WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady          bool   `json:"weighted_admission_resonance_graft_admission_writer_inventory_ready"`
	WeightedAdmissionResonanceGraftAdmissionWriterPreflightConsumed       bool   `json:"weighted_admission_resonance_graft_admission_writer_preflight_consumed"`
	WeightedAdmissionResonanceGraftAdmissionWriterPreflightRequired       bool   `json:"weighted_admission_resonance_graft_admission_writer_preflight_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionWriterInventory          bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_inventory"`
	WeightedAdmissionResonanceGraftAdmissionWriterInventoryID             string `json:"weighted_admission_resonance_graft_admission_writer_inventory_id"`
	ReceiptShape                                                          string `json:"receipt_shape"`
	WriterInventoryKind                                                   string `json:"writer_inventory_kind"`
	WriterInventoryMode                                                   string `json:"writer_inventory_mode"`
	WriterInventoryStage                                                  string `json:"writer_inventory_stage"`
	CausalID                                                              string `json:"causal_id"`
	WriterInventoryHash                                                   string `json:"writer_inventory_hash"`
	ReadBackHash                                                          string `json:"read_back_hash"`
	WriterPreflightVerified                                               bool   `json:"writer_preflight_verified"`
	WriterPreflightHashVerified                                           bool   `json:"writer_preflight_hash_verified"`
	WriterPreflightReadBackVerified                                       bool   `json:"writer_preflight_read_back_verified"`
	LiveStageVerified                                                     bool   `json:"live_stage_verified"`
	LiveStageHashVerified                                                 bool   `json:"live_stage_hash_verified"`
	LiveStageReadBackVerified                                             bool   `json:"live_stage_read_back_verified"`
	EnableGateVerified                                                    bool   `json:"enable_gate_verified"`
	EnableGateHashVerified                                                bool   `json:"enable_gate_hash_verified"`
	EnableGateReadBackVerified                                            bool   `json:"enable_gate_read_back_verified"`
	SwitchVerified                                                        bool   `json:"switch_verified"`
	SwitchHashVerified                                                    bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                                bool   `json:"switch_read_back_verified"`
	PromotionVerified                                                     bool   `json:"promotion_verified"`
	PromotionHashVerified                                                 bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                             bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                      bool   `json:"decision_verified"`
	DecisionHashVerified                                                  bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                              bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                             bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                              bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                          bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                         bool   `json:"proof_verified"`
	ProofHashVerified                                                     bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                 bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                   bool   `json:"store_reader_verified"`
	StoreVerified                                                         bool   `json:"store_verified"`
	CandidateVerified                                                     bool   `json:"candidate_verified"`
	GateVerified                                                          bool   `json:"gate_verified"`
	PreflightVerified                                                     bool   `json:"preflight_verified"`
	BoundaryVerified                                                      bool   `json:"boundary_verified"`
	ObservationVerified                                                   bool   `json:"observation_verified"`
	ReceiverVerified                                                      bool   `json:"receiver_verified"`
	IntentVerified                                                        bool   `json:"intent_verified"`
	FinalGateVerified                                                     bool   `json:"final_gate_verified"`
	SealVerified                                                          bool   `json:"seal_verified"`
	PermitVerified                                                        bool   `json:"permit_verified"`
	AuthorityVerified                                                     bool   `json:"authority_verified"`
	AdmissionRequired                                                     bool   `json:"admission_required"`
	ShadowOnly                                                            bool   `json:"shadow_only"`
	GraftAllowed                                                          bool   `json:"graft_allowed"`
	DryRunOnly                                                            bool   `json:"dry_run_only"`
	LiveReady                                                             bool   `json:"live_ready"`
	RawDreamTextAllowed                                                   bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                  bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                 bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                   bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                   bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                   bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                   bool   `json:"body_mutation_allowed"`
	RequiresWriter                                                        bool   `json:"requires_writer"`
	WriterReady                                                           bool   `json:"writer_ready"`
	RollbackRequired                                                      bool   `json:"rollback_required"`
	RequiresRollback                                                      bool   `json:"requires_rollback"`
	RollbackReady                                                         bool   `json:"rollback_ready"`
	ReadOnly                                                              bool   `json:"read_only"`
	ReplayOnly                                                            bool   `json:"replay_only"`
	SourceSchema                                                          string `json:"source_schema"`
	SourceStatus                                                          string `json:"source_status"`
	SourceTarget                                                          string `json:"source_target"`
	SourceReport                                                          string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID       string `json:"source_weighted_admission_resonance_graft_admission_writer_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady    bool   `json:"source_weighted_admission_resonance_graft_admission_writer_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID string `json:"source_weighted_admission_resonance_graft_admission_writer_preflight_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash     string `json:"source_weighted_admission_resonance_graft_admission_writer_preflight_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack string `json:"source_weighted_admission_resonance_graft_admission_writer_preflight_read_back_hash"`
	SourceWriterPreflightReceiptShape                                     string `json:"source_writer_preflight_receipt_shape"`
	SourceWriterPreflightKind                                             string `json:"source_writer_preflight_kind"`
	SourceWriterPreflightMode                                             string `json:"source_writer_preflight_mode"`
	SourceWriterPreflightStage                                            string `json:"source_writer_preflight_stage"`
	SourceWriterPreflightWriterState                                      string `json:"source_writer_preflight_writer_state"`
	SourceWriterPreflightWriterAction                                     string `json:"source_writer_preflight_writer_action"`
	SourceWriterPreflightRollbackState                                    string `json:"source_writer_preflight_rollback_state"`
	SourceWriterPreflightRollbackAction                                   string `json:"source_writer_preflight_rollback_action"`
	SourceWriterPreflightAdmissionRequired                                bool   `json:"source_writer_preflight_admission_required"`
	SourceWriterPreflightShadowOnly                                       bool   `json:"source_writer_preflight_shadow_only"`
	SourceWriterPreflightGraftAllowed                                     bool   `json:"source_writer_preflight_graft_allowed"`
	SourceWriterPreflightDryRunOnly                                       bool   `json:"source_writer_preflight_dry_run_only"`
	SourceWriterPreflightLiveReady                                        bool   `json:"source_writer_preflight_live_ready"`
	SourceWriterPreflightRawDreamTextAllowed                              bool   `json:"source_writer_preflight_raw_dream_text_allowed"`
	SourceWriterPreflightRawDreamTextObserved                             bool   `json:"source_writer_preflight_raw_dream_text_observed"`
	SourceWriterPreflightRawDreamTextForwarded                            bool   `json:"source_writer_preflight_raw_dream_text_forwarded"`
	SourceWriterPreflightJanusSurfaceAllowed                              bool   `json:"source_writer_preflight_janus_surface_allowed"`
	SourceWriterPreflightCoocLearningAllowed                              bool   `json:"source_writer_preflight_cooc_learning_allowed"`
	SourceWriterPreflightDeltaHarvestAllowed                              bool   `json:"source_writer_preflight_delta_harvest_allowed"`
	SourceWriterPreflightBodyMutationAllowed                              bool   `json:"source_writer_preflight_body_mutation_allowed"`
	SourceWriterPreflightRequiresWriter                                   bool   `json:"source_writer_preflight_requires_writer"`
	SourceWriterPreflightWriterReady                                      bool   `json:"source_writer_preflight_writer_ready"`
	SourceWriterPreflightRollbackRequired                                 bool   `json:"source_writer_preflight_rollback_required"`
	SourceWriterPreflightRequiresRollback                                 bool   `json:"source_writer_preflight_requires_rollback"`
	SourceWriterPreflightRollbackReady                                    bool   `json:"source_writer_preflight_rollback_ready"`
	SourceWriterPreflightReadOnly                                         bool   `json:"source_writer_preflight_read_only"`
	SourceWriterPreflightReplayOnly                                       bool   `json:"source_writer_preflight_replay_only"`
	SourceWriterPreflightWriteAllowed                                     bool   `json:"source_writer_preflight_write_allowed"`
	SourceWriterPreflightAdmissionAllowed                                 bool   `json:"source_writer_preflight_admission_allowed"`
	SourceWriterPreflightLiveAdmissionEnabled                             bool   `json:"source_writer_preflight_live_admission_enabled"`
	SourceWriterPreflightMutatesState                                     bool   `json:"source_writer_preflight_mutates_state"`
	SourceWriterPreflightBodyTarget                                       string `json:"source_writer_preflight_body_target"`
	SourceWriterPreflightPassed                                           bool   `json:"source_writer_preflight_passed"`
	SourceWriterPreflightReason                                           string `json:"source_writer_preflight_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID             string `json:"source_weighted_admission_resonance_graft_admission_live_stage_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady          bool   `json:"source_weighted_admission_resonance_graft_admission_live_stage_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID       string `json:"source_weighted_admission_resonance_graft_admission_live_stage_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash           string `json:"source_weighted_admission_resonance_graft_admission_live_stage_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack       string `json:"source_weighted_admission_resonance_graft_admission_live_stage_read_back_hash"`
	SourceStageState                                                      string `json:"source_stage_state"`
	SourceStageAction                                                     string `json:"source_stage_action"`
	SourceLiveStageReceiptShape                                           string `json:"source_live_stage_receipt_shape"`
	SourceLiveStageKind                                                   string `json:"source_live_stage_kind"`
	SourceLiveStageMode                                                   string `json:"source_live_stage_mode"`
	SourceLiveStageStage                                                  string `json:"source_live_stage_stage"`
	SourceLiveStageAdmissionRequired                                      bool   `json:"source_live_stage_admission_required"`
	SourceLiveStageShadowOnly                                             bool   `json:"source_live_stage_shadow_only"`
	SourceLiveStageGraftAllowed                                           bool   `json:"source_live_stage_graft_allowed"`
	SourceLiveStageDryRunOnly                                             bool   `json:"source_live_stage_dry_run_only"`
	SourceLiveStageLiveReady                                              bool   `json:"source_live_stage_live_ready"`
	SourceLiveStageRawDreamTextAllowed                                    bool   `json:"source_live_stage_raw_dream_text_allowed"`
	SourceLiveStageRawDreamTextObserved                                   bool   `json:"source_live_stage_raw_dream_text_observed"`
	SourceLiveStageRawDreamTextForwarded                                  bool   `json:"source_live_stage_raw_dream_text_forwarded"`
	SourceLiveStageJanusSurfaceAllowed                                    bool   `json:"source_live_stage_janus_surface_allowed"`
	SourceLiveStageCoocLearningAllowed                                    bool   `json:"source_live_stage_cooc_learning_allowed"`
	SourceLiveStageDeltaHarvestAllowed                                    bool   `json:"source_live_stage_delta_harvest_allowed"`
	SourceLiveStageBodyMutationAllowed                                    bool   `json:"source_live_stage_body_mutation_allowed"`
	SourceLiveStageRequiresWriter                                         bool   `json:"source_live_stage_requires_writer"`
	SourceLiveStageWriterReady                                            bool   `json:"source_live_stage_writer_ready"`
	SourceLiveStageRollbackRequired                                       bool   `json:"source_live_stage_rollback_required"`
	SourceLiveStageRequiresRollback                                       bool   `json:"source_live_stage_requires_rollback"`
	SourceLiveStageRollbackReady                                          bool   `json:"source_live_stage_rollback_ready"`
	SourceLiveStageReadOnly                                               bool   `json:"source_live_stage_read_only"`
	SourceLiveStageReplayOnly                                             bool   `json:"source_live_stage_replay_only"`
	SourceLiveStageWriteAllowed                                           bool   `json:"source_live_stage_write_allowed"`
	SourceLiveStageAdmissionAllowed                                       bool   `json:"source_live_stage_admission_allowed"`
	SourceLiveStageLiveAdmissionEnabled                                   bool   `json:"source_live_stage_live_admission_enabled"`
	SourceLiveStageMutatesState                                           bool   `json:"source_live_stage_mutates_state"`
	SourceLiveStageBodyTarget                                             string `json:"source_live_stage_body_target"`
	SourceLiveStagePassed                                                 bool   `json:"source_live_stage_passed"`
	SourceLiveStageReason                                                 string `json:"source_live_stage_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID            string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady         bool   `json:"source_weighted_admission_resonance_graft_admission_enable_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID      string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash          string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack      string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash"`
	SourceEnableState                                                     string `json:"source_enable_state"`
	SourceEnableAction                                                    string `json:"source_enable_action"`
	SourceEnableGateReceiptShape                                          string `json:"source_enable_gate_receipt_shape"`
	SourceEnableGateKind                                                  string `json:"source_enable_gate_kind"`
	SourceEnableGateMode                                                  string `json:"source_enable_gate_mode"`
	SourceEnableGateStage                                                 string `json:"source_enable_gate_stage"`
	SourceEnableGateAdmissionRequired                                     bool   `json:"source_enable_gate_admission_required"`
	SourceEnableGateShadowOnly                                            bool   `json:"source_enable_gate_shadow_only"`
	SourceEnableGateGraftAllowed                                          bool   `json:"source_enable_gate_graft_allowed"`
	SourceEnableGateDryRunOnly                                            bool   `json:"source_enable_gate_dry_run_only"`
	SourceEnableGateLiveReady                                             bool   `json:"source_enable_gate_live_ready"`
	SourceEnableGateRawDreamTextAllowed                                   bool   `json:"source_enable_gate_raw_dream_text_allowed"`
	SourceEnableGateRawDreamTextObserved                                  bool   `json:"source_enable_gate_raw_dream_text_observed"`
	SourceEnableGateRawDreamTextForwarded                                 bool   `json:"source_enable_gate_raw_dream_text_forwarded"`
	SourceEnableGateJanusSurfaceAllowed                                   bool   `json:"source_enable_gate_janus_surface_allowed"`
	SourceEnableGateCoocLearningAllowed                                   bool   `json:"source_enable_gate_cooc_learning_allowed"`
	SourceEnableGateDeltaHarvestAllowed                                   bool   `json:"source_enable_gate_delta_harvest_allowed"`
	SourceEnableGateBodyMutationAllowed                                   bool   `json:"source_enable_gate_body_mutation_allowed"`
	SourceEnableGateRollbackRequired                                      bool   `json:"source_enable_gate_rollback_required"`
	SourceEnableGateReadOnly                                              bool   `json:"source_enable_gate_read_only"`
	SourceEnableGateReplayOnly                                            bool   `json:"source_enable_gate_replay_only"`
	SourceEnableGateWriteAllowed                                          bool   `json:"source_enable_gate_write_allowed"`
	SourceEnableGateAdmissionAllowed                                      bool   `json:"source_enable_gate_admission_allowed"`
	SourceEnableGateLiveAdmissionEnabled                                  bool   `json:"source_enable_gate_live_admission_enabled"`
	SourceEnableGateMutatesState                                          bool   `json:"source_enable_gate_mutates_state"`
	SourceEnableGateBodyTarget                                            string `json:"source_enable_gate_body_target"`
	SourceEnableGatePassed                                                bool   `json:"source_enable_gate_passed"`
	SourceEnableGateReason                                                string `json:"source_enable_gate_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchID                string `json:"source_weighted_admission_resonance_graft_admission_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady             bool   `json:"source_weighted_admission_resonance_graft_admission_switch_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID          string `json:"source_weighted_admission_resonance_graft_admission_switch_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash              string `json:"source_weighted_admission_resonance_graft_admission_switch_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack          string `json:"source_weighted_admission_resonance_graft_admission_switch_read_back_hash"`
	SourceSwitchState                                                     string `json:"source_switch_state"`
	SourceSwitchAction                                                    string `json:"source_switch_action"`
	SourceSwitchReceiptShape                                              string `json:"source_switch_receipt_shape"`
	SourceSwitchKind                                                      string `json:"source_switch_kind"`
	SourceSwitchMode                                                      string `json:"source_switch_mode"`
	SourceSwitchStage                                                     string `json:"source_switch_stage"`
	SourceSwitchAdmissionRequired                                         bool   `json:"source_switch_admission_required"`
	SourceSwitchShadowOnly                                                bool   `json:"source_switch_shadow_only"`
	SourceSwitchGraftAllowed                                              bool   `json:"source_switch_graft_allowed"`
	SourceSwitchDryRunOnly                                                bool   `json:"source_switch_dry_run_only"`
	SourceSwitchLiveReady                                                 bool   `json:"source_switch_live_ready"`
	SourceSwitchRawDreamTextAllowed                                       bool   `json:"source_switch_raw_dream_text_allowed"`
	SourceSwitchRawDreamTextObserved                                      bool   `json:"source_switch_raw_dream_text_observed"`
	SourceSwitchRawDreamTextForwarded                                     bool   `json:"source_switch_raw_dream_text_forwarded"`
	SourceSwitchJanusSurfaceAllowed                                       bool   `json:"source_switch_janus_surface_allowed"`
	SourceSwitchCoocLearningAllowed                                       bool   `json:"source_switch_cooc_learning_allowed"`
	SourceSwitchDeltaHarvestAllowed                                       bool   `json:"source_switch_delta_harvest_allowed"`
	SourceSwitchBodyMutationAllowed                                       bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchRollbackRequired                                          bool   `json:"source_switch_rollback_required"`
	SourceSwitchReadOnly                                                  bool   `json:"source_switch_read_only"`
	SourceSwitchReplayOnly                                                bool   `json:"source_switch_replay_only"`
	SourceSwitchWriteAllowed                                              bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                          bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                      bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchMutatesState                                              bool   `json:"source_switch_mutates_state"`
	SourceSwitchBodyTarget                                                string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                    bool   `json:"source_switch_passed"`
	SourceSwitchReason                                                    string `json:"source_switch_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionID             string `json:"source_weighted_admission_resonance_graft_admission_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady          bool   `json:"source_weighted_admission_resonance_graft_admission_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID       string `json:"source_weighted_admission_resonance_graft_admission_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash           string `json:"source_weighted_admission_resonance_graft_admission_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack       string `json:"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash"`
	SourcePromotion                                                       string `json:"source_promotion"`
	SourcePromotionAction                                                 string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                           string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                   string `json:"source_promotion_kind"`
	SourcePromotionMode                                                   string `json:"source_promotion_mode"`
	SourcePromotionStage                                                  string `json:"source_promotion_stage"`
	SourcePromotionAdmissionRequired                                      bool   `json:"source_promotion_admission_required"`
	SourcePromotionShadowOnly                                             bool   `json:"source_promotion_shadow_only"`
	SourcePromotionGraftAllowed                                           bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionDryRunOnly                                             bool   `json:"source_promotion_dry_run_only"`
	SourcePromotionLiveReady                                              bool   `json:"source_promotion_live_ready"`
	SourcePromotionRawDreamTextAllowed                                    bool   `json:"source_promotion_raw_dream_text_allowed"`
	SourcePromotionRawDreamTextObserved                                   bool   `json:"source_promotion_raw_dream_text_observed"`
	SourcePromotionRawDreamTextForwarded                                  bool   `json:"source_promotion_raw_dream_text_forwarded"`
	SourcePromotionJanusSurfaceAllowed                                    bool   `json:"source_promotion_janus_surface_allowed"`
	SourcePromotionCoocLearningAllowed                                    bool   `json:"source_promotion_cooc_learning_allowed"`
	SourcePromotionDeltaHarvestAllowed                                    bool   `json:"source_promotion_delta_harvest_allowed"`
	SourcePromotionBodyMutationAllowed                                    bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionRollbackRequired                                       bool   `json:"source_promotion_rollback_required"`
	SourcePromotionReadOnly                                               bool   `json:"source_promotion_read_only"`
	SourcePromotionReplayOnly                                             bool   `json:"source_promotion_replay_only"`
	SourcePromotionWriteAllowed                                           bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                       bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                   bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionMutatesState                                           bool   `json:"source_promotion_mutates_state"`
	SourcePromotionBodyTarget                                             string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                 bool   `json:"source_promotion_passed"`
	SourcePromotionReason                                                 string `json:"source_promotion_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionID              string `json:"source_weighted_admission_resonance_graft_admission_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady           bool   `json:"source_weighted_admission_resonance_graft_admission_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID     string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady  bool   `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofID                 string `json:"source_weighted_admission_resonance_graft_admission_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReady              bool   `json:"source_weighted_admission_resonance_graft_admission_proof_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID           string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady        bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID                 string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady              bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateID                      string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady                   bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftGateID                           string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                        bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightID                      string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady                   bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                       string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady                    bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                         string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                      bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                            string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                         bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                           bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                                 bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                      bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                                    bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                              bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                              bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                                     bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                      bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                   bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                          bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                               bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                                bool   `json:"source_authority_granted"`
	AuthorityGranted                                                      bool   `json:"authority_granted"`
	ContractsReady                                                        bool   `json:"contracts_ready"`
	WriteAllowed                                                          bool   `json:"write_allowed"`
	AdmissionAllowed                                                      bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                  bool   `json:"live_admission_enabled"`
	MutatesState                                                          bool   `json:"mutates_state"`
	BodyTarget                                                            string `json:"body_target"`
	Passed                                                                bool   `json:"passed"`
	Reason                                                                string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventory(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT")
	}
	writerPreflightPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory output path missing")
	}
	sourcePreflight, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportForAssert(writerPreflightPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportError(sourcePreflight, root); err != nil {
		return err
	}
	inventory := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport{
		Schema:                  admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema,
		Status:                  "shadow_graft_admission_writer_inventory_blocked_dry_run",
		Target:                  "live_route_admission_next_step",
		TargetKind:              "weighted_internal_world_shadow_graft_admission_writer_inventory",
		TargetMode:              "closed_writer_inventory_guard_dry_run",
		Action:                  "block_weighted_resonance_shadow_graft_admission_writer_preflight_blocked_dry_run",
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
		ReceiptShape:            "weighted_resonance_shadow_graft_admission_writer_inventory_receipt",
		WriterInventoryKind:     "shadow_graft_admission_writer_inventory",
		WriterInventoryMode:     "closed_writer_preflight_inventory_guard",
		WriterInventoryStage:    "pre_writer_contract_graft_admission_writer_inventory",
		WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady:    true,
		WeightedAdmissionResonanceGraftAdmissionWriterPreflightConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionWriterPreflightRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionWriterInventory:    true,
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
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID:       sourcePreflight.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady:    sourcePreflight.WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID: sourcePreflight.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash:     sourcePreflight.WriterPreflightHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack: sourcePreflight.ReadBackHash,
		SourceWriterPreflightReceiptShape:                                     sourcePreflight.ReceiptShape,
		SourceWriterPreflightKind:                                             sourcePreflight.WriterPreflightKind,
		SourceWriterPreflightMode:                                             sourcePreflight.WriterPreflightMode,
		SourceWriterPreflightStage:                                            sourcePreflight.WriterPreflightStage,
		SourceWriterPreflightWriterState:                                      sourcePreflight.WriterState,
		SourceWriterPreflightWriterAction:                                     sourcePreflight.WriterAction,
		SourceWriterPreflightRollbackState:                                    sourcePreflight.RollbackState,
		SourceWriterPreflightRollbackAction:                                   sourcePreflight.RollbackAction,
		SourceWriterPreflightAdmissionRequired:                                sourcePreflight.AdmissionRequired,
		SourceWriterPreflightShadowOnly:                                       sourcePreflight.ShadowOnly,
		SourceWriterPreflightGraftAllowed:                                     sourcePreflight.GraftAllowed,
		SourceWriterPreflightDryRunOnly:                                       sourcePreflight.DryRunOnly,
		SourceWriterPreflightLiveReady:                                        sourcePreflight.LiveReady,
		SourceWriterPreflightRawDreamTextAllowed:                              sourcePreflight.RawDreamTextAllowed,
		SourceWriterPreflightRawDreamTextObserved:                             sourcePreflight.RawDreamTextObserved,
		SourceWriterPreflightRawDreamTextForwarded:                            sourcePreflight.RawDreamTextForwarded,
		SourceWriterPreflightJanusSurfaceAllowed:                              sourcePreflight.JanusSurfaceAllowed,
		SourceWriterPreflightCoocLearningAllowed:                              sourcePreflight.CoocLearningAllowed,
		SourceWriterPreflightDeltaHarvestAllowed:                              sourcePreflight.DeltaHarvestAllowed,
		SourceWriterPreflightBodyMutationAllowed:                              sourcePreflight.BodyMutationAllowed,
		SourceWriterPreflightRequiresWriter:                                   sourcePreflight.RequiresWriter,
		SourceWriterPreflightWriterReady:                                      sourcePreflight.WriterReady,
		SourceWriterPreflightRollbackRequired:                                 sourcePreflight.RollbackRequired,
		SourceWriterPreflightRequiresRollback:                                 sourcePreflight.RequiresRollback,
		SourceWriterPreflightRollbackReady:                                    sourcePreflight.RollbackReady,
		SourceWriterPreflightReadOnly:                                         sourcePreflight.ReadOnly,
		SourceWriterPreflightReplayOnly:                                       sourcePreflight.ReplayOnly,
		SourceWriterPreflightWriteAllowed:                                     sourcePreflight.WriteAllowed,
		SourceWriterPreflightAdmissionAllowed:                                 sourcePreflight.AdmissionAllowed,
		SourceWriterPreflightLiveAdmissionEnabled:                             sourcePreflight.LiveAdmissionEnabled,
		SourceWriterPreflightMutatesState:                                     sourcePreflight.MutatesState,
		SourceWriterPreflightBodyTarget:                                       sourcePreflight.BodyTarget,
		SourceWriterPreflightPassed:                                           sourcePreflight.Passed,
		SourceWriterPreflightReason:                                           sourcePreflight.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID:             sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady:          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash:           sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceStageState:                                                      sourcePreflight.SourceStageState,
		SourceStageAction:                                                     sourcePreflight.SourceStageAction,
		SourceLiveStageReceiptShape:                                           sourcePreflight.SourceLiveStageReceiptShape,
		SourceLiveStageKind:                                                   sourcePreflight.SourceLiveStageKind,
		SourceLiveStageMode:                                                   sourcePreflight.SourceLiveStageMode,
		SourceLiveStageStage:                                                  sourcePreflight.SourceLiveStageStage,
		SourceLiveStageAdmissionRequired:                                      sourcePreflight.SourceLiveStageAdmissionRequired,
		SourceLiveStageShadowOnly:                                             sourcePreflight.SourceLiveStageShadowOnly,
		SourceLiveStageGraftAllowed:                                           sourcePreflight.SourceLiveStageGraftAllowed,
		SourceLiveStageDryRunOnly:                                             sourcePreflight.SourceLiveStageDryRunOnly,
		SourceLiveStageLiveReady:                                              sourcePreflight.SourceLiveStageLiveReady,
		SourceLiveStageRawDreamTextAllowed:                                    sourcePreflight.SourceLiveStageRawDreamTextAllowed,
		SourceLiveStageRawDreamTextObserved:                                   sourcePreflight.SourceLiveStageRawDreamTextObserved,
		SourceLiveStageRawDreamTextForwarded:                                  sourcePreflight.SourceLiveStageRawDreamTextForwarded,
		SourceLiveStageJanusSurfaceAllowed:                                    sourcePreflight.SourceLiveStageJanusSurfaceAllowed,
		SourceLiveStageCoocLearningAllowed:                                    sourcePreflight.SourceLiveStageCoocLearningAllowed,
		SourceLiveStageDeltaHarvestAllowed:                                    sourcePreflight.SourceLiveStageDeltaHarvestAllowed,
		SourceLiveStageBodyMutationAllowed:                                    sourcePreflight.SourceLiveStageBodyMutationAllowed,
		SourceLiveStageRequiresWriter:                                         sourcePreflight.SourceLiveStageRequiresWriter,
		SourceLiveStageWriterReady:                                            sourcePreflight.SourceLiveStageWriterReady,
		SourceLiveStageRollbackRequired:                                       sourcePreflight.SourceLiveStageRollbackRequired,
		SourceLiveStageRequiresRollback:                                       sourcePreflight.SourceLiveStageRequiresRollback,
		SourceLiveStageRollbackReady:                                          sourcePreflight.SourceLiveStageRollbackReady,
		SourceLiveStageReadOnly:                                               sourcePreflight.SourceLiveStageReadOnly,
		SourceLiveStageReplayOnly:                                             sourcePreflight.SourceLiveStageReplayOnly,
		SourceLiveStageWriteAllowed:                                           sourcePreflight.SourceLiveStageWriteAllowed,
		SourceLiveStageAdmissionAllowed:                                       sourcePreflight.SourceLiveStageAdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:                                   sourcePreflight.SourceLiveStageLiveAdmissionEnabled,
		SourceLiveStageMutatesState:                                           sourcePreflight.SourceLiveStageMutatesState,
		SourceLiveStageBodyTarget:                                             sourcePreflight.SourceLiveStageBodyTarget,
		SourceLiveStagePassed:                                                 sourcePreflight.SourceLiveStagePassed,
		SourceLiveStageReason:                                                 sourcePreflight.SourceLiveStageReason,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID:            sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady:         sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID:      sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash:          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack:      sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceEnableState:                                                     sourcePreflight.SourceEnableState,
		SourceEnableAction:                                                    sourcePreflight.SourceEnableAction,
		SourceEnableGateReceiptShape:                                          sourcePreflight.SourceEnableGateReceiptShape,
		SourceEnableGateKind:                                                  sourcePreflight.SourceEnableGateKind,
		SourceEnableGateMode:                                                  sourcePreflight.SourceEnableGateMode,
		SourceEnableGateStage:                                                 sourcePreflight.SourceEnableGateStage,
		SourceEnableGateAdmissionRequired:                                     sourcePreflight.SourceEnableGateAdmissionRequired,
		SourceEnableGateShadowOnly:                                            sourcePreflight.SourceEnableGateShadowOnly,
		SourceEnableGateGraftAllowed:                                          sourcePreflight.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:                                            sourcePreflight.SourceEnableGateDryRunOnly,
		SourceEnableGateLiveReady:                                             sourcePreflight.SourceEnableGateLiveReady,
		SourceEnableGateRawDreamTextAllowed:                                   sourcePreflight.SourceEnableGateRawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:                                  sourcePreflight.SourceEnableGateRawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded:                                 sourcePreflight.SourceEnableGateRawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:                                   sourcePreflight.SourceEnableGateJanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:                                   sourcePreflight.SourceEnableGateCoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:                                   sourcePreflight.SourceEnableGateDeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:                                   sourcePreflight.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:                                      sourcePreflight.SourceEnableGateRollbackRequired,
		SourceEnableGateReadOnly:                                              sourcePreflight.SourceEnableGateReadOnly,
		SourceEnableGateReplayOnly:                                            sourcePreflight.SourceEnableGateReplayOnly,
		SourceEnableGateWriteAllowed:                                          sourcePreflight.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:                                      sourcePreflight.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:                                  sourcePreflight.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:                                          sourcePreflight.SourceEnableGateMutatesState,
		SourceEnableGateBodyTarget:                                            sourcePreflight.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                                                sourcePreflight.SourceEnableGatePassed,
		SourceEnableGateReason:                                                sourcePreflight.SourceEnableGateReason,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:                sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:             sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID:          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack:          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourceSwitchState:                                                     sourcePreflight.SourceSwitchState,
		SourceSwitchAction:                                                    sourcePreflight.SourceSwitchAction,
		SourceSwitchReceiptShape:                                              sourcePreflight.SourceSwitchReceiptShape,
		SourceSwitchKind:                                                      sourcePreflight.SourceSwitchKind,
		SourceSwitchMode:                                                      sourcePreflight.SourceSwitchMode,
		SourceSwitchStage:                                                     sourcePreflight.SourceSwitchStage,
		SourceSwitchAdmissionRequired:                                         sourcePreflight.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:                                                sourcePreflight.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:                                              sourcePreflight.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:                                                sourcePreflight.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:                                                 sourcePreflight.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:                                       sourcePreflight.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                                      sourcePreflight.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                                     sourcePreflight.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                                       sourcePreflight.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                                       sourcePreflight.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                                       sourcePreflight.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                                       sourcePreflight.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:                                          sourcePreflight.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:                                                  sourcePreflight.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:                                                sourcePreflight.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:                                              sourcePreflight.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:                                          sourcePreflight.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                                      sourcePreflight.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:                                              sourcePreflight.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:                                                sourcePreflight.SourceSwitchBodyTarget,
		SourceSwitchPassed:                                                    sourcePreflight.SourceSwitchPassed,
		SourceSwitchReason:                                                    sourcePreflight.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:             sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:          sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:           sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack:       sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                       sourcePreflight.SourcePromotion,
		SourcePromotionAction:                                                 sourcePreflight.SourcePromotionAction,
		SourcePromotionReceiptShape:                                           sourcePreflight.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                   sourcePreflight.SourcePromotionKind,
		SourcePromotionMode:                                                   sourcePreflight.SourcePromotionMode,
		SourcePromotionStage:                                                  sourcePreflight.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                      sourcePreflight.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                             sourcePreflight.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                           sourcePreflight.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                             sourcePreflight.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                              sourcePreflight.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                    sourcePreflight.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                   sourcePreflight.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                  sourcePreflight.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                    sourcePreflight.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                    sourcePreflight.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                    sourcePreflight.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                    sourcePreflight.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                       sourcePreflight.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                               sourcePreflight.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                             sourcePreflight.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                           sourcePreflight.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                       sourcePreflight.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                   sourcePreflight.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                           sourcePreflight.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                             sourcePreflight.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                 sourcePreflight.SourcePromotionPassed,
		SourcePromotionReason:                                                 sourcePreflight.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:           sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:     sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady:  sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                 sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:              sourcePreflight.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:           sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:        sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                 sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:              sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                      sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                   sourcePreflight.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                           sourcePreflight.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                        sourcePreflight.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                      sourcePreflight.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                   sourcePreflight.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                       sourcePreflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                    sourcePreflight.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                         sourcePreflight.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                      sourcePreflight.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                            sourcePreflight.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                         sourcePreflight.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                           sourcePreflight.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                 sourcePreflight.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                      sourcePreflight.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                    sourcePreflight.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                              sourcePreflight.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                              sourcePreflight.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                     sourcePreflight.BodySmokeWeighted,
		NanoDirectRunner:                                                      sourcePreflight.NanoDirectRunner,
		NanoDirectFinalGate:                                                   sourcePreflight.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                          sourcePreflight.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                               sourcePreflight.BoundaryReportFullChain,
		SourceAuthorityGranted:                                                sourcePreflight.SourceAuthorityGranted,
		AuthorityGranted:                                                      false,
		ContractsReady:                                                        false,
		WriteAllowed:                                                          false,
		AdmissionAllowed:                                                      false,
		LiveAdmissionEnabled:                                                  false,
		MutatesState:                                                          false,
		BodyTarget:                                                            "none",
		Passed:                                                                true,
		Reason:                                                                "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent",
	}
	inventory.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID(inventory)
	inventory.WriterInventoryHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash(inventory)
	inventory.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBackHash(inventory)
	inventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryID(inventory)
	if inventory.CausalID == "" ||
		inventory.WriterInventoryHash == "" ||
		inventory.ReadBackHash == "" ||
		inventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID == "" ||
		inventory.WriterInventoryHash == inventory.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory read-back proof failed")
	}
	raw, err := json.MarshalIndent(inventory, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory] pass: resonance_graft_admission_writer_inventory_report=%s resonance_graft_admission_writer_preflight_report=%s\n", outputPath, writerPreflightPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-inventory-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema)
	}
	if report.Status != "shadow_graft_admission_writer_inventory_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory status mismatch: got %q want %q", report.Status, "shadow_graft_admission_writer_inventory_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_writer_inventory")
	}
	if report.TargetMode != "closed_writer_inventory_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory target_mode mismatch: got %q want %q", report.TargetMode, "closed_writer_inventory_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_writer_preflight_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_writer_preflight")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_writer_preflight")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.InventoryState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory inventory_state mismatch: got %q want %q", report.InventoryState, "blocked")
	}
	if report.InventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory inventory_action mismatch: got %q want %q", report.InventoryAction, "reject_blocked_writer_preflight")
	}
	if report.WriterContract != "none" || report.RollbackContract != "none" || report.AdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory contracts unexpectedly named")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_writer_inventory_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_writer_inventory_receipt")
	}
	if report.WriterInventoryKind != "shadow_graft_admission_writer_inventory" ||
		report.WriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		report.WriterInventoryStage != "pre_writer_contract_graft_admission_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_writer_inventory_ready", report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady},
		{"weighted_admission_resonance_graft_admission_writer_preflight_consumed", report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightConsumed},
		{"weighted_admission_resonance_graft_admission_writer_preflight_required", report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightRequired},
		{"next_step_blocked_without_resonance_graft_admission_writer_inventory", report.NextStepBlockedWithoutResonanceGraftAdmissionWriterInventory},
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
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady},
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
		{"source_weighted_admission_resonance_graft_admission_live_stage_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady},
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
		{"source_weighted_admission_resonance_graft_admission_enable_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady},
		{"source_enable_gate_admission_required", report.SourceEnableGateAdmissionRequired},
		{"source_enable_gate_shadow_only", report.SourceEnableGateShadowOnly},
		{"source_enable_gate_dry_run_only", report.SourceEnableGateDryRunOnly},
		{"source_enable_gate_live_ready", report.SourceEnableGateLiveReady},
		{"source_enable_gate_rollback_required", report.SourceEnableGateRollbackRequired},
		{"source_enable_gate_read_only", report.SourceEnableGateReadOnly},
		{"source_enable_gate_replay_only", report.SourceEnableGateReplayOnly},
		{"source_enable_gate_passed", report.SourceEnableGatePassed},
		{"source_weighted_admission_resonance_graft_admission_switch_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady},
		{"source_switch_admission_required", report.SourceSwitchAdmissionRequired},
		{"source_switch_shadow_only", report.SourceSwitchShadowOnly},
		{"source_switch_dry_run_only", report.SourceSwitchDryRunOnly},
		{"source_switch_live_ready", report.SourceSwitchLiveReady},
		{"source_switch_rollback_required", report.SourceSwitchRollbackRequired},
		{"source_switch_read_only", report.SourceSwitchReadOnly},
		{"source_switch_replay_only", report.SourceSwitchReplayOnly},
		{"source_switch_passed", report.SourceSwitchPassed},
		{"source_weighted_admission_resonance_graft_admission_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady},
		{"source_promotion_admission_required", report.SourcePromotionAdmissionRequired},
		{"source_promotion_shadow_only", report.SourcePromotionShadowOnly},
		{"source_promotion_dry_run_only", report.SourcePromotionDryRunOnly},
		{"source_promotion_live_ready", report.SourcePromotionLiveReady},
		{"source_promotion_rollback_required", report.SourcePromotionRollbackRequired},
		{"source_promotion_read_only", report.SourcePromotionReadOnly},
		{"source_promotion_replay_only", report.SourcePromotionReplayOnly},
		{"source_promotion_passed", report.SourcePromotionPassed},
		{"source_weighted_admission_resonance_graft_admission_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady},
		{"source_weighted_admission_resonance_graft_admission_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReady},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady},
		{"source_weighted_admission_resonance_graft_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReady},
		{"source_weighted_admission_resonance_graft_candidate_ready", report.SourceWeightedAdmissionResonanceGraftCandidateReady},
		{"source_weighted_admission_resonance_graft_gate_ready", report.SourceWeightedAdmissionResonanceGraftGateReady},
		{"source_weighted_admission_resonance_graft_preflight_ready", report.SourceWeightedAdmissionResonanceGraftPreflightReady},
		{"source_weighted_admission_resonance_graft_boundary_ready", report.SourceWeightedAdmissionResonanceGraftBoundaryReady},
		{"source_weighted_admission_resonance_observation_ready", report.SourceWeightedAdmissionResonanceObservationReady},
		{"source_weighted_admission_resonance_receiver_ready", report.SourceWeightedAdmissionResonanceReceiverReady},
		{"source_weighted_admission_resonance_intent_ready", report.SourceWeightedAdmissionResonanceIntentReady},
		{"source_weighted_admission_final_gate_ready", report.SourceWeightedAdmissionFinalGateReady},
		{"source_weighted_admission_seal_ready", report.SourceWeightedAdmissionSealReady},
		{"source_weighted_admission_permit_ready", report.SourceWeightedAdmissionPermitReady},
		{"source_weighted_admission_authority_consumed", report.SourceWeightedAdmissionAuthorityConsumed},
		{"source_weighted_admission_authority_required", report.SourceWeightedAdmissionAuthorityRequired},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission writer inventory %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission writer inventory opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_writer_inventory_id", report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID},
		{"causal_id", report.CausalID},
		{"writer_inventory_hash", report.WriterInventoryHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID},
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID},
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash},
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack},
		{"source_weighted_admission_resonance_graft_admission_live_stage_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID},
		{"source_weighted_admission_resonance_graft_admission_live_stage_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID},
		{"source_weighted_admission_resonance_graft_admission_live_stage_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash},
		{"source_weighted_admission_resonance_graft_admission_live_stage_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack},
		{"source_weighted_admission_resonance_graft_admission_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID},
		{"source_weighted_admission_resonance_graft_admission_switch_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID},
		{"source_weighted_admission_resonance_graft_admission_switch_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash},
		{"source_weighted_admission_resonance_graft_admission_switch_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack},
		{"source_weighted_admission_resonance_graft_admission_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID},
		{"source_weighted_admission_resonance_graft_admission_promotion_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID},
		{"source_weighted_admission_resonance_graft_admission_promotion_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash},
		{"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack},
		{"source_weighted_admission_resonance_graft_admission_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofID},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreID},
		{"source_weighted_admission_resonance_graft_candidate_id", report.SourceWeightedAdmissionResonanceGraftCandidateID},
		{"source_weighted_admission_resonance_graft_gate_id", report.SourceWeightedAdmissionResonanceGraftGateID},
		{"source_weighted_admission_resonance_graft_preflight_id", report.SourceWeightedAdmissionResonanceGraftPreflightID},
		{"source_weighted_admission_resonance_graft_boundary_id", report.SourceWeightedAdmissionResonanceGraftBoundaryID},
		{"source_weighted_admission_resonance_observation_id", report.SourceWeightedAdmissionResonanceObservationID},
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission writer inventory %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_writer_preflight_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceWriterPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_writer_preflight_receipt" ||
		report.SourceWriterPreflightKind != "shadow_graft_admission_writer_preflight" ||
		report.SourceWriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.SourceWriterPreflightStage != "pre_writer_inventory_graft_admission_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source writer preflight shape mismatch")
	}
	if report.SourceWriterPreflightWriterState != "blocked" ||
		report.SourceWriterPreflightWriterAction != "reject_blocked_live_stage" ||
		report.SourceWriterPreflightRollbackState != "blocked" ||
		report.SourceWriterPreflightRollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source writer preflight state mismatch")
	}
	if report.SourceWriterPreflightBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_writer_preflight_body_target mismatch: got %q want %q", report.SourceWriterPreflightBodyTarget, "none")
	}
	if report.SourceWriterPreflightReason != "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_writer_preflight_reason mismatch: got %q", report.SourceWriterPreflightReason)
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_live_stage" ||
		report.SourceLiveStageMode != "closed_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "pre_writer_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		report.SourceEnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.SourceEnableGateMode != "closed_switch_enable_guard" ||
		report.SourceEnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID, "weighted-resonance-graft-admission-writer-inventory-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-writer-inventory-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory causal prefix mismatch")
	}
	if !strings.HasPrefix(report.WriterInventoryHash, "weighted-resonance-graft-admission-writer-inventory-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-writer-inventory-read-") ||
		report.WriterInventoryHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID, "weighted-resonance-graft-admission-writer-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID, "weighted-resonance-graft-admission-writer-preflight-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash, "weighted-resonance-graft-admission-writer-preflight-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack, "weighted-resonance-graft-admission-writer-preflight-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source writer preflight mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID, "weighted-resonance-graft-admission-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID, "weighted-resonance-graft-admission-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash, "weighted-resonance-graft-admission-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack, "weighted-resonance-graft-admission-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID, "weighted-resonance-graft-admission-enable-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash, "weighted-resonance-graft-admission-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack, "weighted-resonance-graft-admission-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory causal_id mismatch")
	}
	if report.WriterInventoryHash == "" || report.WriterInventoryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory writer_inventory_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory read_back_hash mismatch")
	}
	if report.WriterInventoryHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID == "" || report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer inventory reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport) string {
	h := hashJSON(struct {
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		SourceLiveStageID         string `json:"source_live_stage_id"`
		Target                    string `json:"target"`
		WriterInventoryKind       string `json:"writer_inventory_kind"`
		WriterInventoryStage      string `json:"writer_inventory_stage"`
	}{
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		SourceLiveStageID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		Target:                    sw.Target,
		WriterInventoryKind:       sw.WriterInventoryKind,
		WriterInventoryStage:      sw.WriterInventoryStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-writer-inventory-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport) string {
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
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
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
	return "weighted-resonance-graft-admission-writer-inventory-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport) string {
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
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		WriterInventoryKind:       sw.WriterInventoryKind,
		WriterInventoryReady:      sw.WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady,
		WriterPreflightConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionWriterPreflightConsumed,
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
	return "weighted-resonance-graft-admission-writer-inventory-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport) string {
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
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_inventory"`
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
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		SourceLiveStageID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceLiveStageHash:       sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceLiveStageRead:       sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceEnableGateID:        sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateHash:      sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceEnableGateRead:      sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceSwitchID:            sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchHash:          sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceSwitchRead:          sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourcePromotionID:         sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceDecisionID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceProofID:             sw.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:            sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:             sw.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:         sw.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:              sw.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:         sw.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:          sw.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:       sw.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:          sw.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                  sw.CausalID,
		WriterInventoryHash:       sw.WriterInventoryHash,
		ReadBackHash:              sw.ReadBackHash,
		Ready:                     sw.WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady,
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
		NextStepBlockedWithout:    sw.NextStepBlockedWithoutResonanceGraftAdmissionWriterInventory,
		SourcePromotionReady:      sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceDecisionReady:       sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourcePreconditionReady:   sw.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceProofReady:          sw.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:         sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:          sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:      sw.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:           sw.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:      sw.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:       sw.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:    sw.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:       sw.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:         sw.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:      sw.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:           sw.SourceWeightedAdmissionSealReady,
		SourcePermitReady:         sw.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:       sw.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:     sw.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-writer-inventory-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer inventory path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission writer inventory not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer inventory not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer inventory JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer inventory decode failed: %w", err)
	}
	return report, root, nil
}
