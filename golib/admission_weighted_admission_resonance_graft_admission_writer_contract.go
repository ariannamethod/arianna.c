package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_writer_contract.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport struct {
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
	ContractState                                                         string `json:"contract_state"`
	ContractAction                                                        string `json:"contract_action"`
	WriterContract                                                        string `json:"writer_contract"`
	RollbackContract                                                      string `json:"rollback_contract"`
	AdmissionLedgerContract                                               string `json:"admission_ledger_contract"`
	WriterContractShape                                                   string `json:"writer_contract_shape"`
	RollbackContractShape                                                 string `json:"rollback_contract_shape"`
	LedgerContractShape                                                   string `json:"ledger_contract_shape"`
	WriteScope                                                            string `json:"write_scope"`
	RollbackScope                                                         string `json:"rollback_scope"`
	LedgerMode                                                            string `json:"ledger_mode"`
	WriterContractPresent                                                 bool   `json:"writer_contract_present"`
	RollbackContractPresent                                               bool   `json:"rollback_contract_present"`
	LedgerContractPresent                                                 bool   `json:"ledger_contract_present"`
	WeightedAdmissionResonanceGraftAdmissionWriterContractReady           bool   `json:"weighted_admission_resonance_graft_admission_writer_contract_ready"`
	WeightedAdmissionResonanceGraftAdmissionWriterInventoryConsumed       bool   `json:"weighted_admission_resonance_graft_admission_writer_inventory_consumed"`
	WeightedAdmissionResonanceGraftAdmissionWriterInventoryRequired       bool   `json:"weighted_admission_resonance_graft_admission_writer_inventory_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionWriterContract           bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_contract"`
	WeightedAdmissionResonanceGraftAdmissionWriterContractID              string `json:"weighted_admission_resonance_graft_admission_writer_contract_id"`
	ReceiptShape                                                          string `json:"receipt_shape"`
	WriterContractKind                                                    string `json:"writer_contract_kind"`
	WriterContractMode                                                    string `json:"writer_contract_mode"`
	WriterContractStage                                                   string `json:"writer_contract_stage"`
	CausalID                                                              string `json:"causal_id"`
	WriterContractHash                                                    string `json:"writer_contract_hash"`
	ReadBackHash                                                          string `json:"read_back_hash"`
	WriterInventoryVerified                                               bool   `json:"writer_inventory_verified"`
	WriterInventoryHashVerified                                           bool   `json:"writer_inventory_hash_verified"`
	WriterInventoryReadBackVerified                                       bool   `json:"writer_inventory_read_back_verified"`
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
	SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID       string `json:"source_weighted_admission_resonance_graft_admission_writer_inventory_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReady    bool   `json:"source_weighted_admission_resonance_graft_admission_writer_inventory_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID string `json:"source_weighted_admission_resonance_graft_admission_writer_inventory_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash     string `json:"source_weighted_admission_resonance_graft_admission_writer_inventory_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack string `json:"source_weighted_admission_resonance_graft_admission_writer_inventory_read_back_hash"`
	SourceWriterInventoryReceiptShape                                     string `json:"source_writer_inventory_receipt_shape"`
	SourceWriterInventoryKind                                             string `json:"source_writer_inventory_kind"`
	SourceWriterInventoryMode                                             string `json:"source_writer_inventory_mode"`
	SourceWriterInventoryStage                                            string `json:"source_writer_inventory_stage"`
	SourceWriterInventoryWriterState                                      string `json:"source_writer_inventory_writer_state"`
	SourceWriterInventoryWriterAction                                     string `json:"source_writer_inventory_writer_action"`
	SourceWriterInventoryRollbackState                                    string `json:"source_writer_inventory_rollback_state"`
	SourceWriterInventoryRollbackAction                                   string `json:"source_writer_inventory_rollback_action"`
	SourceWriterInventoryInventoryState                                   string `json:"source_writer_inventory_inventory_state"`
	SourceWriterInventoryInventoryAction                                  string `json:"source_writer_inventory_inventory_action"`
	SourceWriterInventoryWriterContract                                   string `json:"source_writer_inventory_writer_contract"`
	SourceWriterInventoryRollbackContract                                 string `json:"source_writer_inventory_rollback_contract"`
	SourceWriterInventoryAdmissionLedgerContract                          string `json:"source_writer_inventory_admission_ledger_contract"`
	SourceWriterInventoryWriterContractPresent                            bool   `json:"source_writer_inventory_writer_contract_present"`
	SourceWriterInventoryRollbackContractPresent                          bool   `json:"source_writer_inventory_rollback_contract_present"`
	SourceWriterInventoryLedgerContractPresent                            bool   `json:"source_writer_inventory_ledger_contract_present"`
	SourceWriterInventoryContractsReady                                   bool   `json:"source_writer_inventory_contracts_ready"`
	SourceWriterInventoryBodyTarget                                       string `json:"source_writer_inventory_body_target"`
	SourceWriterInventoryReason                                           string `json:"source_writer_inventory_reason"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContract(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-contract RESONANCE_GRAFT_ADMISSION_WRITER_INVENTORY_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_CONTRACT_REPORT")
	}
	writerInventoryPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract output path missing")
	}
	sourceInventory, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportForAssert(writerInventoryPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventoryReportError(sourceInventory, root); err != nil {
		return err
	}
	inventory := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport{
		Schema:                  admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema,
		Status:                  "shadow_graft_admission_writer_contract_blocked_dry_run",
		Target:                  "live_route_admission_next_step",
		TargetKind:              "weighted_internal_world_shadow_graft_admission_writer_contract",
		TargetMode:              "closed_writer_contract_guard_dry_run",
		Action:                  "block_weighted_resonance_shadow_graft_admission_writer_inventory_blocked_dry_run",
		WriterState:             "blocked",
		WriterAction:            "reject_blocked_writer_inventory",
		RollbackState:           "blocked",
		RollbackAction:          "reject_blocked_writer_inventory",
		StageState:              sourceInventory.StageState,
		StageAction:             sourceInventory.StageAction,
		EnableState:             sourceInventory.EnableState,
		EnableAction:            sourceInventory.EnableAction,
		SwitchState:             sourceInventory.SwitchState,
		SwitchAction:            sourceInventory.SwitchAction,
		Promotion:               sourceInventory.Promotion,
		InventoryState:          "blocked",
		InventoryAction:         "reject_blocked_writer_preflight",
		ContractState:           "blocked",
		ContractAction:          "reject_blocked_writer_inventory",
		WriterContract:          "none",
		RollbackContract:        "none",
		AdmissionLedgerContract: "none",
		WriterContractShape:     "none",
		RollbackContractShape:   "none",
		LedgerContractShape:     "none",
		WriteScope:              "none",
		RollbackScope:           "none",
		LedgerMode:              "none",
		ReceiptShape:            "weighted_resonance_shadow_graft_admission_writer_contract_receipt",
		WriterContractKind:      "shadow_graft_admission_writer_contract",
		WriterContractMode:      "closed_writer_inventory_contract_guard",
		WriterContractStage:     "pre_admission_ledger_graft_admission_writer_contract",
		WeightedAdmissionResonanceGraftAdmissionWriterContractReady:     true,
		WeightedAdmissionResonanceGraftAdmissionWriterInventoryConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionWriterInventoryRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionWriterContract:     true,
		WriterInventoryVerified:         true,
		WriterInventoryHashVerified:     true,
		WriterInventoryReadBackVerified: true,
		WriterPreflightVerified:         true,
		WriterPreflightHashVerified:     true,
		WriterPreflightReadBackVerified: true,
		LiveStageVerified:               sourceInventory.LiveStageVerified,
		LiveStageHashVerified:           sourceInventory.LiveStageHashVerified,
		LiveStageReadBackVerified:       sourceInventory.LiveStageReadBackVerified,
		EnableGateVerified:              sourceInventory.EnableGateVerified,
		EnableGateHashVerified:          sourceInventory.EnableGateHashVerified,
		EnableGateReadBackVerified:      sourceInventory.EnableGateReadBackVerified,
		SwitchVerified:                  sourceInventory.SwitchVerified,
		SwitchHashVerified:              sourceInventory.SwitchHashVerified,
		SwitchReadBackVerified:          sourceInventory.SwitchReadBackVerified,
		PromotionVerified:               sourceInventory.PromotionVerified,
		PromotionHashVerified:           sourceInventory.PromotionHashVerified,
		PromotionReadBackVerified:       sourceInventory.PromotionReadBackVerified,
		DecisionVerified:                sourceInventory.DecisionVerified,
		DecisionHashVerified:            sourceInventory.DecisionHashVerified,
		DecisionReadBackVerified:        sourceInventory.DecisionReadBackVerified,
		ProofPreconditionVerified:       sourceInventory.ProofPreconditionVerified,
		PreconditionHashVerified:        sourceInventory.PreconditionHashVerified,
		PreconditionReadBackVerified:    sourceInventory.PreconditionReadBackVerified,
		ProofVerified:                   sourceInventory.ProofVerified,
		ProofHashVerified:               sourceInventory.ProofHashVerified,
		ProofReadBackVerified:           sourceInventory.ProofReadBackVerified,
		StoreReaderVerified:             sourceInventory.StoreReaderVerified,
		StoreVerified:                   sourceInventory.StoreVerified,
		CandidateVerified:               sourceInventory.CandidateVerified,
		GateVerified:                    sourceInventory.GateVerified,
		PreflightVerified:               sourceInventory.PreflightVerified,
		BoundaryVerified:                sourceInventory.BoundaryVerified,
		ObservationVerified:             sourceInventory.ObservationVerified,
		ReceiverVerified:                sourceInventory.ReceiverVerified,
		IntentVerified:                  sourceInventory.IntentVerified,
		FinalGateVerified:               sourceInventory.FinalGateVerified,
		SealVerified:                    sourceInventory.SealVerified,
		PermitVerified:                  sourceInventory.PermitVerified,
		AuthorityVerified:               sourceInventory.AuthorityVerified,
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
		SourceSchema:                    sourceInventory.Schema,
		SourceStatus:                    sourceInventory.Status,
		SourceTarget:                    sourceInventory.Target,
		SourceReport:                    writerInventoryPath,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID:       sourceInventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReady:    sourceInventory.WeightedAdmissionResonanceGraftAdmissionWriterInventoryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID: sourceInventory.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash:     sourceInventory.WriterInventoryHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack: sourceInventory.ReadBackHash,
		SourceWriterInventoryReceiptShape:                                     sourceInventory.ReceiptShape,
		SourceWriterInventoryKind:                                             sourceInventory.WriterInventoryKind,
		SourceWriterInventoryMode:                                             sourceInventory.WriterInventoryMode,
		SourceWriterInventoryStage:                                            sourceInventory.WriterInventoryStage,
		SourceWriterInventoryWriterState:                                      sourceInventory.WriterState,
		SourceWriterInventoryWriterAction:                                     sourceInventory.WriterAction,
		SourceWriterInventoryRollbackState:                                    sourceInventory.RollbackState,
		SourceWriterInventoryRollbackAction:                                   sourceInventory.RollbackAction,
		SourceWriterInventoryInventoryState:                                   sourceInventory.InventoryState,
		SourceWriterInventoryInventoryAction:                                  sourceInventory.InventoryAction,
		SourceWriterInventoryWriterContract:                                   sourceInventory.WriterContract,
		SourceWriterInventoryRollbackContract:                                 sourceInventory.RollbackContract,
		SourceWriterInventoryAdmissionLedgerContract:                          sourceInventory.AdmissionLedgerContract,
		SourceWriterInventoryWriterContractPresent:                            sourceInventory.WriterContractPresent,
		SourceWriterInventoryRollbackContractPresent:                          sourceInventory.RollbackContractPresent,
		SourceWriterInventoryLedgerContractPresent:                            sourceInventory.LedgerContractPresent,
		SourceWriterInventoryContractsReady:                                   sourceInventory.ContractsReady,
		SourceWriterInventoryBodyTarget:                                       sourceInventory.BodyTarget,
		SourceWriterInventoryReason:                                           sourceInventory.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID:       sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady:    sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID: sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash:     sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack: sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		SourceWriterPreflightReceiptShape:                                     sourceInventory.SourceWriterPreflightReceiptShape,
		SourceWriterPreflightKind:                                             sourceInventory.SourceWriterPreflightKind,
		SourceWriterPreflightMode:                                             sourceInventory.SourceWriterPreflightMode,
		SourceWriterPreflightStage:                                            sourceInventory.SourceWriterPreflightStage,
		SourceWriterPreflightWriterState:                                      sourceInventory.SourceWriterPreflightWriterState,
		SourceWriterPreflightWriterAction:                                     sourceInventory.SourceWriterPreflightWriterAction,
		SourceWriterPreflightRollbackState:                                    sourceInventory.SourceWriterPreflightRollbackState,
		SourceWriterPreflightRollbackAction:                                   sourceInventory.SourceWriterPreflightRollbackAction,
		SourceWriterPreflightAdmissionRequired:                                sourceInventory.SourceWriterPreflightAdmissionRequired,
		SourceWriterPreflightShadowOnly:                                       sourceInventory.SourceWriterPreflightShadowOnly,
		SourceWriterPreflightGraftAllowed:                                     sourceInventory.SourceWriterPreflightGraftAllowed,
		SourceWriterPreflightDryRunOnly:                                       sourceInventory.SourceWriterPreflightDryRunOnly,
		SourceWriterPreflightLiveReady:                                        sourceInventory.SourceWriterPreflightLiveReady,
		SourceWriterPreflightRawDreamTextAllowed:                              sourceInventory.SourceWriterPreflightRawDreamTextAllowed,
		SourceWriterPreflightRawDreamTextObserved:                             sourceInventory.SourceWriterPreflightRawDreamTextObserved,
		SourceWriterPreflightRawDreamTextForwarded:                            sourceInventory.SourceWriterPreflightRawDreamTextForwarded,
		SourceWriterPreflightJanusSurfaceAllowed:                              sourceInventory.SourceWriterPreflightJanusSurfaceAllowed,
		SourceWriterPreflightCoocLearningAllowed:                              sourceInventory.SourceWriterPreflightCoocLearningAllowed,
		SourceWriterPreflightDeltaHarvestAllowed:                              sourceInventory.SourceWriterPreflightDeltaHarvestAllowed,
		SourceWriterPreflightBodyMutationAllowed:                              sourceInventory.SourceWriterPreflightBodyMutationAllowed,
		SourceWriterPreflightRequiresWriter:                                   sourceInventory.SourceWriterPreflightRequiresWriter,
		SourceWriterPreflightWriterReady:                                      sourceInventory.SourceWriterPreflightWriterReady,
		SourceWriterPreflightRollbackRequired:                                 sourceInventory.SourceWriterPreflightRollbackRequired,
		SourceWriterPreflightRequiresRollback:                                 sourceInventory.SourceWriterPreflightRequiresRollback,
		SourceWriterPreflightRollbackReady:                                    sourceInventory.SourceWriterPreflightRollbackReady,
		SourceWriterPreflightReadOnly:                                         sourceInventory.SourceWriterPreflightReadOnly,
		SourceWriterPreflightReplayOnly:                                       sourceInventory.SourceWriterPreflightReplayOnly,
		SourceWriterPreflightWriteAllowed:                                     sourceInventory.SourceWriterPreflightWriteAllowed,
		SourceWriterPreflightAdmissionAllowed:                                 sourceInventory.SourceWriterPreflightAdmissionAllowed,
		SourceWriterPreflightLiveAdmissionEnabled:                             sourceInventory.SourceWriterPreflightLiveAdmissionEnabled,
		SourceWriterPreflightMutatesState:                                     sourceInventory.SourceWriterPreflightMutatesState,
		SourceWriterPreflightBodyTarget:                                       sourceInventory.SourceWriterPreflightBodyTarget,
		SourceWriterPreflightPassed:                                           sourceInventory.SourceWriterPreflightPassed,
		SourceWriterPreflightReason:                                           sourceInventory.SourceWriterPreflightReason,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID:             sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady:          sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID:       sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash:           sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack:       sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceStageState:                                                      sourceInventory.SourceStageState,
		SourceStageAction:                                                     sourceInventory.SourceStageAction,
		SourceLiveStageReceiptShape:                                           sourceInventory.SourceLiveStageReceiptShape,
		SourceLiveStageKind:                                                   sourceInventory.SourceLiveStageKind,
		SourceLiveStageMode:                                                   sourceInventory.SourceLiveStageMode,
		SourceLiveStageStage:                                                  sourceInventory.SourceLiveStageStage,
		SourceLiveStageAdmissionRequired:                                      sourceInventory.SourceLiveStageAdmissionRequired,
		SourceLiveStageShadowOnly:                                             sourceInventory.SourceLiveStageShadowOnly,
		SourceLiveStageGraftAllowed:                                           sourceInventory.SourceLiveStageGraftAllowed,
		SourceLiveStageDryRunOnly:                                             sourceInventory.SourceLiveStageDryRunOnly,
		SourceLiveStageLiveReady:                                              sourceInventory.SourceLiveStageLiveReady,
		SourceLiveStageRawDreamTextAllowed:                                    sourceInventory.SourceLiveStageRawDreamTextAllowed,
		SourceLiveStageRawDreamTextObserved:                                   sourceInventory.SourceLiveStageRawDreamTextObserved,
		SourceLiveStageRawDreamTextForwarded:                                  sourceInventory.SourceLiveStageRawDreamTextForwarded,
		SourceLiveStageJanusSurfaceAllowed:                                    sourceInventory.SourceLiveStageJanusSurfaceAllowed,
		SourceLiveStageCoocLearningAllowed:                                    sourceInventory.SourceLiveStageCoocLearningAllowed,
		SourceLiveStageDeltaHarvestAllowed:                                    sourceInventory.SourceLiveStageDeltaHarvestAllowed,
		SourceLiveStageBodyMutationAllowed:                                    sourceInventory.SourceLiveStageBodyMutationAllowed,
		SourceLiveStageRequiresWriter:                                         sourceInventory.SourceLiveStageRequiresWriter,
		SourceLiveStageWriterReady:                                            sourceInventory.SourceLiveStageWriterReady,
		SourceLiveStageRollbackRequired:                                       sourceInventory.SourceLiveStageRollbackRequired,
		SourceLiveStageRequiresRollback:                                       sourceInventory.SourceLiveStageRequiresRollback,
		SourceLiveStageRollbackReady:                                          sourceInventory.SourceLiveStageRollbackReady,
		SourceLiveStageReadOnly:                                               sourceInventory.SourceLiveStageReadOnly,
		SourceLiveStageReplayOnly:                                             sourceInventory.SourceLiveStageReplayOnly,
		SourceLiveStageWriteAllowed:                                           sourceInventory.SourceLiveStageWriteAllowed,
		SourceLiveStageAdmissionAllowed:                                       sourceInventory.SourceLiveStageAdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:                                   sourceInventory.SourceLiveStageLiveAdmissionEnabled,
		SourceLiveStageMutatesState:                                           sourceInventory.SourceLiveStageMutatesState,
		SourceLiveStageBodyTarget:                                             sourceInventory.SourceLiveStageBodyTarget,
		SourceLiveStagePassed:                                                 sourceInventory.SourceLiveStagePassed,
		SourceLiveStageReason:                                                 sourceInventory.SourceLiveStageReason,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID:            sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady:         sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID:      sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash:          sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack:      sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceEnableState:                                                     sourceInventory.SourceEnableState,
		SourceEnableAction:                                                    sourceInventory.SourceEnableAction,
		SourceEnableGateReceiptShape:                                          sourceInventory.SourceEnableGateReceiptShape,
		SourceEnableGateKind:                                                  sourceInventory.SourceEnableGateKind,
		SourceEnableGateMode:                                                  sourceInventory.SourceEnableGateMode,
		SourceEnableGateStage:                                                 sourceInventory.SourceEnableGateStage,
		SourceEnableGateAdmissionRequired:                                     sourceInventory.SourceEnableGateAdmissionRequired,
		SourceEnableGateShadowOnly:                                            sourceInventory.SourceEnableGateShadowOnly,
		SourceEnableGateGraftAllowed:                                          sourceInventory.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:                                            sourceInventory.SourceEnableGateDryRunOnly,
		SourceEnableGateLiveReady:                                             sourceInventory.SourceEnableGateLiveReady,
		SourceEnableGateRawDreamTextAllowed:                                   sourceInventory.SourceEnableGateRawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:                                  sourceInventory.SourceEnableGateRawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded:                                 sourceInventory.SourceEnableGateRawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:                                   sourceInventory.SourceEnableGateJanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:                                   sourceInventory.SourceEnableGateCoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:                                   sourceInventory.SourceEnableGateDeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:                                   sourceInventory.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:                                      sourceInventory.SourceEnableGateRollbackRequired,
		SourceEnableGateReadOnly:                                              sourceInventory.SourceEnableGateReadOnly,
		SourceEnableGateReplayOnly:                                            sourceInventory.SourceEnableGateReplayOnly,
		SourceEnableGateWriteAllowed:                                          sourceInventory.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:                                      sourceInventory.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:                                  sourceInventory.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:                                          sourceInventory.SourceEnableGateMutatesState,
		SourceEnableGateBodyTarget:                                            sourceInventory.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                                                sourceInventory.SourceEnableGatePassed,
		SourceEnableGateReason:                                                sourceInventory.SourceEnableGateReason,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:                sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:             sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID:          sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:              sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack:          sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourceSwitchState:                                                     sourceInventory.SourceSwitchState,
		SourceSwitchAction:                                                    sourceInventory.SourceSwitchAction,
		SourceSwitchReceiptShape:                                              sourceInventory.SourceSwitchReceiptShape,
		SourceSwitchKind:                                                      sourceInventory.SourceSwitchKind,
		SourceSwitchMode:                                                      sourceInventory.SourceSwitchMode,
		SourceSwitchStage:                                                     sourceInventory.SourceSwitchStage,
		SourceSwitchAdmissionRequired:                                         sourceInventory.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:                                                sourceInventory.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:                                              sourceInventory.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:                                                sourceInventory.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:                                                 sourceInventory.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:                                       sourceInventory.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                                      sourceInventory.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                                     sourceInventory.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                                       sourceInventory.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                                       sourceInventory.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                                       sourceInventory.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                                       sourceInventory.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:                                          sourceInventory.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:                                                  sourceInventory.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:                                                sourceInventory.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:                                              sourceInventory.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:                                          sourceInventory.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                                      sourceInventory.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:                                              sourceInventory.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:                                                sourceInventory.SourceSwitchBodyTarget,
		SourceSwitchPassed:                                                    sourceInventory.SourceSwitchPassed,
		SourceSwitchReason:                                                    sourceInventory.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:             sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:          sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID:       sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:           sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack:       sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                       sourceInventory.SourcePromotion,
		SourcePromotionAction:                                                 sourceInventory.SourcePromotionAction,
		SourcePromotionReceiptShape:                                           sourceInventory.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                   sourceInventory.SourcePromotionKind,
		SourcePromotionMode:                                                   sourceInventory.SourcePromotionMode,
		SourcePromotionStage:                                                  sourceInventory.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                      sourceInventory.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                             sourceInventory.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                           sourceInventory.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                             sourceInventory.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                              sourceInventory.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                    sourceInventory.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                   sourceInventory.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                  sourceInventory.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                    sourceInventory.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                    sourceInventory.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                    sourceInventory.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                    sourceInventory.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                       sourceInventory.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                               sourceInventory.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                             sourceInventory.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                           sourceInventory.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                       sourceInventory.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                   sourceInventory.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                           sourceInventory.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                             sourceInventory.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                 sourceInventory.SourcePromotionPassed,
		SourcePromotionReason:                                                 sourceInventory.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:              sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:           sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:     sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady:  sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                 sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:              sourceInventory.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:           sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:        sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                 sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:              sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                      sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                   sourceInventory.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                           sourceInventory.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                        sourceInventory.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                      sourceInventory.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                   sourceInventory.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                       sourceInventory.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                    sourceInventory.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                         sourceInventory.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                      sourceInventory.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                            sourceInventory.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                         sourceInventory.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                           sourceInventory.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                 sourceInventory.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                      sourceInventory.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                    sourceInventory.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                              sourceInventory.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                              sourceInventory.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                     sourceInventory.BodySmokeWeighted,
		NanoDirectRunner:                                                      sourceInventory.NanoDirectRunner,
		NanoDirectFinalGate:                                                   sourceInventory.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                          sourceInventory.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                               sourceInventory.BoundaryReportFullChain,
		SourceAuthorityGranted:                                                sourceInventory.SourceAuthorityGranted,
		AuthorityGranted:                                                      false,
		ContractsReady:                                                        false,
		WriteAllowed:                                                          false,
		AdmissionAllowed:                                                      false,
		LiveAdmissionEnabled:                                                  false,
		MutatesState:                                                          false,
		BodyTarget:                                                            "none",
		Passed:                                                                true,
		Reason:                                                                "weighted resonance shadow graft admission writer contract blocked by blocked writer inventory; writer, rollback, and ledger contract shapes remain absent",
	}
	inventory.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID(inventory)
	inventory.WriterContractHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractHash(inventory)
	inventory.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReadBackHash(inventory)
	inventory.WeightedAdmissionResonanceGraftAdmissionWriterContractID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractID(inventory)
	if inventory.CausalID == "" ||
		inventory.WriterContractHash == "" ||
		inventory.ReadBackHash == "" ||
		inventory.WeightedAdmissionResonanceGraftAdmissionWriterContractID == "" ||
		inventory.WriterContractHash == inventory.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer contract read-back proof failed")
	}
	raw, err := json.MarshalIndent(inventory, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer contract marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer contract write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-writer-contract] pass: resonance_graft_admission_writer_contract_report=%s resonance_graft_admission_writer_inventory_report=%s\n", outputPath, writerInventoryPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-contract-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission writer contract schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema {
		return fmt.Errorf("weighted admission resonance graft admission writer contract schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema)
	}
	if report.Status != "shadow_graft_admission_writer_contract_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract status mismatch: got %q want %q", report.Status, "shadow_graft_admission_writer_contract_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_writer_contract")
	}
	if report.TargetMode != "closed_writer_contract_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract target_mode mismatch: got %q want %q", report.TargetMode, "closed_writer_contract_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_writer_inventory_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_writer_inventory_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_writer_inventory")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_writer_inventory")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.InventoryState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract inventory_state mismatch: got %q want %q", report.InventoryState, "blocked")
	}
	if report.InventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract inventory_action mismatch: got %q want %q", report.InventoryAction, "reject_blocked_writer_preflight")
	}
	if report.ContractState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract contract_state mismatch: got %q want %q", report.ContractState, "blocked")
	}
	if report.ContractAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract contract_action mismatch: got %q want %q", report.ContractAction, "reject_blocked_writer_inventory")
	}
	if report.WriterContract != "none" || report.RollbackContract != "none" || report.AdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract contracts unexpectedly named")
	}
	if report.WriterContractShape != "none" || report.RollbackContractShape != "none" || report.LedgerContractShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract contract shapes unexpectedly named")
	}
	if report.WriteScope != "none" || report.RollbackScope != "none" || report.LedgerMode != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract scopes unexpectedly opened")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_writer_contract_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_writer_contract_receipt")
	}
	if report.WriterContractKind != "shadow_graft_admission_writer_contract" ||
		report.WriterContractMode != "closed_writer_inventory_contract_guard" ||
		report.WriterContractStage != "pre_admission_ledger_graft_admission_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_writer_contract_ready", report.WeightedAdmissionResonanceGraftAdmissionWriterContractReady},
		{"weighted_admission_resonance_graft_admission_writer_inventory_consumed", report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryConsumed},
		{"weighted_admission_resonance_graft_admission_writer_inventory_required", report.WeightedAdmissionResonanceGraftAdmissionWriterInventoryRequired},
		{"next_step_blocked_without_resonance_graft_admission_writer_contract", report.NextStepBlockedWithoutResonanceGraftAdmissionWriterContract},
		{"writer_inventory_verified", report.WriterInventoryVerified},
		{"writer_inventory_hash_verified", report.WriterInventoryHashVerified},
		{"writer_inventory_read_back_verified", report.WriterInventoryReadBackVerified},
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
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReady},
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
			return fmt.Errorf("weighted admission resonance graft admission writer contract %s not ready", required.name)
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
		{"source_writer_inventory_writer_contract_present", report.SourceWriterInventoryWriterContractPresent},
		{"source_writer_inventory_rollback_contract_present", report.SourceWriterInventoryRollbackContractPresent},
		{"source_writer_inventory_ledger_contract_present", report.SourceWriterInventoryLedgerContractPresent},
		{"source_writer_inventory_contracts_ready", report.SourceWriterInventoryContractsReady},
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
			return fmt.Errorf("weighted admission resonance graft admission writer contract opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_writer_contract_id", report.WeightedAdmissionResonanceGraftAdmissionWriterContractID},
		{"causal_id", report.CausalID},
		{"writer_contract_hash", report.WriterContractHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID},
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID},
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash},
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack},
		{"source_writer_inventory_reason", report.SourceWriterInventoryReason},
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
			return fmt.Errorf("weighted admission resonance graft admission writer contract %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterInventorySchema)
	}
	if report.SourceStatus != "shadow_graft_admission_writer_inventory_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_writer_inventory_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceWriterInventoryReceiptShape != "weighted_resonance_shadow_graft_admission_writer_inventory_receipt" ||
		report.SourceWriterInventoryKind != "shadow_graft_admission_writer_inventory" ||
		report.SourceWriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		report.SourceWriterInventoryStage != "pre_writer_contract_graft_admission_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer inventory shape mismatch")
	}
	if report.SourceWriterInventoryWriterState != "blocked" ||
		report.SourceWriterInventoryWriterAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryRollbackState != "blocked" ||
		report.SourceWriterInventoryRollbackAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryInventoryState != "blocked" ||
		report.SourceWriterInventoryInventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer inventory state mismatch")
	}
	if report.SourceWriterInventoryWriterContract != "none" ||
		report.SourceWriterInventoryRollbackContract != "none" ||
		report.SourceWriterInventoryAdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer inventory contracts unexpectedly named")
	}
	if report.SourceWriterInventoryBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_writer_inventory_body_target mismatch: got %q want %q", report.SourceWriterInventoryBodyTarget, "none")
	}
	if report.SourceWriterInventoryReason != "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_writer_inventory_reason mismatch: got %q", report.SourceWriterInventoryReason)
	}
	if report.SourceWriterPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_writer_preflight_receipt" ||
		report.SourceWriterPreflightKind != "shadow_graft_admission_writer_preflight" ||
		report.SourceWriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.SourceWriterPreflightStage != "pre_writer_inventory_graft_admission_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer preflight shape mismatch")
	}
	if report.SourceWriterPreflightWriterState != "blocked" ||
		report.SourceWriterPreflightWriterAction != "reject_blocked_live_stage" ||
		report.SourceWriterPreflightRollbackState != "blocked" ||
		report.SourceWriterPreflightRollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer preflight state mismatch")
	}
	if report.SourceWriterPreflightBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_writer_preflight_body_target mismatch: got %q want %q", report.SourceWriterPreflightBodyTarget, "none")
	}
	if report.SourceWriterPreflightReason != "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_writer_preflight_reason mismatch: got %q", report.SourceWriterPreflightReason)
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_live_stage" ||
		report.SourceLiveStageMode != "closed_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "pre_writer_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		report.SourceEnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.SourceEnableGateMode != "closed_switch_enable_guard" ||
		report.SourceEnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionWriterContractID, "weighted-resonance-graft-admission-writer-contract-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-writer-contract-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract causal prefix mismatch")
	}
	if !strings.HasPrefix(report.WriterContractHash, "weighted-resonance-graft-admission-writer-contract-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-writer-contract-read-") ||
		report.WriterContractHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer contract read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID, "weighted-resonance-graft-admission-writer-inventory-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID, "weighted-resonance-graft-admission-writer-inventory-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash, "weighted-resonance-graft-admission-writer-inventory-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack, "weighted-resonance-graft-admission-writer-inventory-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer inventory mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID, "weighted-resonance-graft-admission-writer-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID, "weighted-resonance-graft-admission-writer-preflight-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash, "weighted-resonance-graft-admission-writer-preflight-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack, "weighted-resonance-graft-admission-writer-preflight-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source writer preflight mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID, "weighted-resonance-graft-admission-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID, "weighted-resonance-graft-admission-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash, "weighted-resonance-graft-admission-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack, "weighted-resonance-graft-admission-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID, "weighted-resonance-graft-admission-enable-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash, "weighted-resonance-graft-admission-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack, "weighted-resonance-graft-admission-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission writer contract source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer contract causal_id mismatch")
	}
	if report.WriterContractHash == "" || report.WriterContractHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer contract writer_contract_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer contract read_back_hash mismatch")
	}
	if report.WriterContractHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer contract read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionWriterContractID == "" || report.WeightedAdmissionResonanceGraftAdmissionWriterContractID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer contract id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission writer contract blocked by blocked writer inventory; writer, rollback, and ledger contract shapes remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer contract reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport) string {
	h := hashJSON(struct {
		SourceWriterInventoryID   string `json:"source_writer_inventory_id"`
		SourceWriterInventoryRead string `json:"source_writer_inventory_read_back_hash"`
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		Target                    string `json:"target"`
		WriterContractKind        string `json:"writer_contract_kind"`
		WriterContractStage       string `json:"writer_contract_stage"`
	}{
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		Target:                    sw.Target,
		WriterContractKind:        sw.WriterContractKind,
		WriterContractStage:       sw.WriterContractStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-writer-contract-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport) string {
	h := hashJSON(struct {
		CausalID                  string `json:"causal_id"`
		SourceWriterInventoryID   string `json:"source_writer_inventory_id"`
		SourceWriterInventoryHash string `json:"source_writer_inventory_hash"`
		SourceWriterInventoryRead string `json:"source_writer_inventory_read_back_hash"`
		SourceWriterPreflightID   string `json:"source_writer_preflight_id"`
		SourceWriterPreflightHash string `json:"source_writer_preflight_hash"`
		SourceWriterPreflightRead string `json:"source_writer_preflight_read_back_hash"`
		WriterState               string `json:"writer_state"`
		WriterAction              string `json:"writer_action"`
		RollbackState             string `json:"rollback_state"`
		RollbackAction            string `json:"rollback_action"`
		InventoryState            string `json:"inventory_state"`
		InventoryAction           string `json:"inventory_action"`
		ContractState             string `json:"contract_state"`
		ContractAction            string `json:"contract_action"`
		WriterContract            string `json:"writer_contract"`
		RollbackContract          string `json:"rollback_contract"`
		AdmissionLedgerContract   string `json:"admission_ledger_contract"`
		WriterContractShape       string `json:"writer_contract_shape"`
		RollbackContractShape     string `json:"rollback_contract_shape"`
		LedgerContractShape       string `json:"ledger_contract_shape"`
		WriteScope                string `json:"write_scope"`
		RollbackScope             string `json:"rollback_scope"`
		LedgerMode                string `json:"ledger_mode"`
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
		WriterContractMode        string `json:"writer_contract_mode"`
		WriterInventoryVerified   bool   `json:"writer_inventory_verified"`
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
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWriterInventoryHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		WriterState:               sw.WriterState,
		WriterAction:              sw.WriterAction,
		RollbackState:             sw.RollbackState,
		RollbackAction:            sw.RollbackAction,
		InventoryState:            sw.InventoryState,
		InventoryAction:           sw.InventoryAction,
		ContractState:             sw.ContractState,
		ContractAction:            sw.ContractAction,
		WriterContract:            sw.WriterContract,
		RollbackContract:          sw.RollbackContract,
		AdmissionLedgerContract:   sw.AdmissionLedgerContract,
		WriterContractShape:       sw.WriterContractShape,
		RollbackContractShape:     sw.RollbackContractShape,
		LedgerContractShape:       sw.LedgerContractShape,
		WriteScope:                sw.WriteScope,
		RollbackScope:             sw.RollbackScope,
		LedgerMode:                sw.LedgerMode,
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
		WriterContractMode:        sw.WriterContractMode,
		WriterInventoryVerified:   sw.WriterInventoryVerified,
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
	return "weighted-resonance-graft-admission-writer-contract-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport) string {
	h := hashJSON(struct {
		WriterContractHash        string `json:"writer_contract_hash"`
		SourceWriterInventoryID   string `json:"source_writer_inventory_id"`
		SourceWriterInventoryRead string `json:"source_writer_inventory_read_back_hash"`
		WriterContractKind        string `json:"writer_contract_kind"`
		WriterContractReady       bool   `json:"writer_contract_ready"`
		WriterInventoryConsumed   bool   `json:"writer_inventory_consumed"`
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
		WriterContractHash:        sw.WriterContractHash,
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack,
		WriterContractKind:        sw.WriterContractKind,
		WriterContractReady:       sw.WeightedAdmissionResonanceGraftAdmissionWriterContractReady,
		WriterInventoryConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionWriterInventoryConsumed,
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
	return "weighted-resonance-graft-admission-writer-contract-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport) string {
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
		ContractState             string `json:"contract_state"`
		ContractAction            string `json:"contract_action"`
		WriterContract            string `json:"writer_contract"`
		RollbackContract          string `json:"rollback_contract"`
		AdmissionLedgerContract   string `json:"admission_ledger_contract"`
		WriterContractShape       string `json:"writer_contract_shape"`
		RollbackContractShape     string `json:"rollback_contract_shape"`
		LedgerContractShape       string `json:"ledger_contract_shape"`
		WriteScope                string `json:"write_scope"`
		RollbackScope             string `json:"rollback_scope"`
		LedgerMode                string `json:"ledger_mode"`
		StageState                string `json:"stage_state"`
		StageAction               string `json:"stage_action"`
		EnableState               string `json:"enable_state"`
		EnableAction              string `json:"enable_action"`
		SwitchState               string `json:"switch_state"`
		SwitchAction              string `json:"switch_action"`
		Promotion                 string `json:"promotion"`
		SourceReport              string `json:"source_report"`
		SourceWriterInventoryID   string `json:"source_writer_inventory_id"`
		SourceWriterInventoryHash string `json:"source_writer_inventory_hash"`
		SourceWriterInventoryRead string `json:"source_writer_inventory_read_back_hash"`
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
		WriterContractHash        string `json:"writer_contract_hash"`
		ReadBackHash              string `json:"read_back_hash"`
		Ready                     bool   `json:"ready"`
		ReceiptShape              string `json:"receipt_shape"`
		WriterContractKind        string `json:"writer_contract_kind"`
		WriterContractMode        string `json:"writer_contract_mode"`
		WriterContractStage       string `json:"writer_contract_stage"`
		WriterInventoryVerified   bool   `json:"writer_inventory_verified"`
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
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_contract"`
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
		ContractState:             sw.ContractState,
		ContractAction:            sw.ContractAction,
		WriterContract:            sw.WriterContract,
		RollbackContract:          sw.RollbackContract,
		AdmissionLedgerContract:   sw.AdmissionLedgerContract,
		WriterContractShape:       sw.WriterContractShape,
		RollbackContractShape:     sw.RollbackContractShape,
		LedgerContractShape:       sw.LedgerContractShape,
		WriteScope:                sw.WriteScope,
		RollbackScope:             sw.RollbackScope,
		LedgerMode:                sw.LedgerMode,
		StageState:                sw.StageState,
		StageAction:               sw.StageAction,
		EnableState:               sw.EnableState,
		EnableAction:              sw.EnableAction,
		SwitchState:               sw.SwitchState,
		SwitchAction:              sw.SwitchAction,
		Promotion:                 sw.Promotion,
		SourceReport:              sw.SourceReport,
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWriterInventoryHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack,
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
		WriterContractHash:        sw.WriterContractHash,
		ReadBackHash:              sw.ReadBackHash,
		Ready:                     sw.WeightedAdmissionResonanceGraftAdmissionWriterContractReady,
		ReceiptShape:              sw.ReceiptShape,
		WriterContractKind:        sw.WriterContractKind,
		WriterContractMode:        sw.WriterContractMode,
		WriterContractStage:       sw.WriterContractStage,
		WriterInventoryVerified:   sw.WriterInventoryVerified,
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
		NextStepBlockedWithout:    sw.NextStepBlockedWithoutResonanceGraftAdmissionWriterContract,
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
	return "weighted-resonance-graft-admission-writer-contract-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer contract path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission writer contract not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer contract not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer contract JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer contract decode failed: %w", err)
	}
	return report, root, nil
}
