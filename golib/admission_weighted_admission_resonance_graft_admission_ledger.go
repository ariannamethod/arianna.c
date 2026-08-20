package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_ledger.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport struct {
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
	LedgerState                                                           string `json:"ledger_state"`
	LedgerAction                                                          string `json:"ledger_action"`
	LedgerContract                                                        string `json:"ledger_contract"`
	LedgerEntrypoint                                                      string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                    string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                      string `json:"ledger_write_scope"`
	LedgerReady                                                           bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                   bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionLedgerReady                   bool   `json:"weighted_admission_resonance_graft_admission_ledger_ready"`
	WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed        bool   `json:"weighted_admission_resonance_graft_admission_writer_contract_consumed"`
	WeightedAdmissionResonanceGraftAdmissionWriterContractRequired        bool   `json:"weighted_admission_resonance_graft_admission_writer_contract_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionLedger                   bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger"`
	WeightedAdmissionResonanceGraftAdmissionLedgerID                      string `json:"weighted_admission_resonance_graft_admission_ledger_id"`
	ReceiptShape                                                          string `json:"receipt_shape"`
	AdmissionLedgerKind                                                   string `json:"admission_ledger_kind"`
	AdmissionLedgerMode                                                   string `json:"admission_ledger_mode"`
	AdmissionLedgerStage                                                  string `json:"admission_ledger_stage"`
	CausalID                                                              string `json:"causal_id"`
	AdmissionLedgerHash                                                   string `json:"admission_ledger_hash"`
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
	SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID        string `json:"source_weighted_admission_resonance_graft_admission_writer_contract_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReady     bool   `json:"source_weighted_admission_resonance_graft_admission_writer_contract_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID  string `json:"source_weighted_admission_resonance_graft_admission_writer_contract_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash      string `json:"source_weighted_admission_resonance_graft_admission_writer_contract_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack  string `json:"source_weighted_admission_resonance_graft_admission_writer_contract_read_back_hash"`
	SourceWriterContractReceiptShape                                      string `json:"source_writer_contract_receipt_shape"`
	SourceWriterContractKind                                              string `json:"source_writer_contract_kind"`
	SourceWriterContractMode                                              string `json:"source_writer_contract_mode"`
	SourceWriterContractStage                                             string `json:"source_writer_contract_stage"`
	SourceWriterContractContractState                                     string `json:"source_writer_contract_contract_state"`
	SourceWriterContractContractAction                                    string `json:"source_writer_contract_contract_action"`
	SourceWriterContractWriterAction                                      string `json:"source_writer_contract_writer_action"`
	SourceWriterContractRollbackAction                                    string `json:"source_writer_contract_rollback_action"`
	SourceWriterContractWriterContract                                    string `json:"source_writer_contract_writer_contract"`
	SourceWriterContractRollbackContract                                  string `json:"source_writer_contract_rollback_contract"`
	SourceWriterContractAdmissionLedgerContract                           string `json:"source_writer_contract_admission_ledger_contract"`
	SourceWriterContractWriterContractShape                               string `json:"source_writer_contract_writer_contract_shape"`
	SourceWriterContractRollbackContractShape                             string `json:"source_writer_contract_rollback_contract_shape"`
	SourceWriterContractLedgerContractShape                               string `json:"source_writer_contract_ledger_contract_shape"`
	SourceWriterContractWriteScope                                        string `json:"source_writer_contract_write_scope"`
	SourceWriterContractRollbackScope                                     string `json:"source_writer_contract_rollback_scope"`
	SourceWriterContractLedgerMode                                        string `json:"source_writer_contract_ledger_mode"`
	SourceWriterContractWriterContractPresent                             bool   `json:"source_writer_contract_writer_contract_present"`
	SourceWriterContractRollbackContractPresent                           bool   `json:"source_writer_contract_rollback_contract_present"`
	SourceWriterContractLedgerContractPresent                             bool   `json:"source_writer_contract_ledger_contract_present"`
	SourceWriterContractContractsReady                                    bool   `json:"source_writer_contract_contracts_ready"`
	SourceWriterContractBodyTarget                                        string `json:"source_writer_contract_body_target"`
	SourceWriterContractReason                                            string `json:"source_writer_contract_reason"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedger(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger RESONANCE_GRAFT_ADMISSION_WRITER_CONTRACT_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_REPORT")
	}
	writerContractPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission ledger output path missing")
	}
	sourceContract, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportForAssert(writerContractPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractReportError(sourceContract, root); err != nil {
		return err
	}
	inventory := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport{
		Schema:                  admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema,
		Status:                  "shadow_graft_admission_ledger_blocked_dry_run",
		Target:                  "live_route_admission_next_step",
		TargetKind:              "weighted_internal_world_shadow_graft_admission_ledger",
		TargetMode:              "closed_admission_ledger_guard_dry_run",
		Action:                  "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run",
		WriterState:             "blocked",
		WriterAction:            "reject_blocked_writer_contract",
		RollbackState:           "blocked",
		RollbackAction:          "reject_blocked_writer_contract",
		StageState:              sourceContract.StageState,
		StageAction:             sourceContract.StageAction,
		EnableState:             sourceContract.EnableState,
		EnableAction:            sourceContract.EnableAction,
		SwitchState:             sourceContract.SwitchState,
		SwitchAction:            sourceContract.SwitchAction,
		Promotion:               sourceContract.Promotion,
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
		LedgerState:             "blocked",
		LedgerAction:            "reject_blocked_writer_contract",
		LedgerContract:          "none",
		LedgerEntrypoint:        "none",
		LedgerReceiptShape:      "none",
		LedgerWriteScope:        "none",
		LedgerReady:             false,
		LedgerAppendAllowed:     false,
		ReceiptShape:            "weighted_resonance_shadow_graft_admission_ledger_receipt",
		AdmissionLedgerKind:     "shadow_graft_admission_ledger",
		AdmissionLedgerMode:     "closed_writer_contract_ledger_guard",
		AdmissionLedgerStage:    "pre_ledger_append_graft_admission_ledger",
		WeightedAdmissionResonanceGraftAdmissionLedgerReady:            true,
		WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionWriterContractRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionLedger:            true,
		WriterInventoryVerified:                                        true,
		WriterInventoryHashVerified:                                    true,
		WriterInventoryReadBackVerified:                                true,
		WriterPreflightVerified:                                        true,
		WriterPreflightHashVerified:                                    true,
		WriterPreflightReadBackVerified:                                true,
		LiveStageVerified:                                              sourceContract.LiveStageVerified,
		LiveStageHashVerified:                                          sourceContract.LiveStageHashVerified,
		LiveStageReadBackVerified:                                      sourceContract.LiveStageReadBackVerified,
		EnableGateVerified:                                             sourceContract.EnableGateVerified,
		EnableGateHashVerified:                                         sourceContract.EnableGateHashVerified,
		EnableGateReadBackVerified:                                     sourceContract.EnableGateReadBackVerified,
		SwitchVerified:                                                 sourceContract.SwitchVerified,
		SwitchHashVerified:                                             sourceContract.SwitchHashVerified,
		SwitchReadBackVerified:                                         sourceContract.SwitchReadBackVerified,
		PromotionVerified:                                              sourceContract.PromotionVerified,
		PromotionHashVerified:                                          sourceContract.PromotionHashVerified,
		PromotionReadBackVerified:                                      sourceContract.PromotionReadBackVerified,
		DecisionVerified:                                               sourceContract.DecisionVerified,
		DecisionHashVerified:                                           sourceContract.DecisionHashVerified,
		DecisionReadBackVerified:                                       sourceContract.DecisionReadBackVerified,
		ProofPreconditionVerified:                                      sourceContract.ProofPreconditionVerified,
		PreconditionHashVerified:                                       sourceContract.PreconditionHashVerified,
		PreconditionReadBackVerified:                                   sourceContract.PreconditionReadBackVerified,
		ProofVerified:                                                  sourceContract.ProofVerified,
		ProofHashVerified:                                              sourceContract.ProofHashVerified,
		ProofReadBackVerified:                                          sourceContract.ProofReadBackVerified,
		StoreReaderVerified:                                            sourceContract.StoreReaderVerified,
		StoreVerified:                                                  sourceContract.StoreVerified,
		CandidateVerified:                                              sourceContract.CandidateVerified,
		GateVerified:                                                   sourceContract.GateVerified,
		PreflightVerified:                                              sourceContract.PreflightVerified,
		BoundaryVerified:                                               sourceContract.BoundaryVerified,
		ObservationVerified:                                            sourceContract.ObservationVerified,
		ReceiverVerified:                                               sourceContract.ReceiverVerified,
		IntentVerified:                                                 sourceContract.IntentVerified,
		FinalGateVerified:                                              sourceContract.FinalGateVerified,
		SealVerified:                                                   sourceContract.SealVerified,
		PermitVerified:                                                 sourceContract.PermitVerified,
		AuthorityVerified:                                              sourceContract.AuthorityVerified,
		AdmissionRequired:                                              true,
		ShadowOnly:                                                     true,
		GraftAllowed:                                                   false,
		DryRunOnly:                                                     true,
		LiveReady:                                                      true,
		RawDreamTextAllowed:                                            false,
		RawDreamTextObserved:                                           false,
		RawDreamTextForwarded:                                          false,
		JanusSurfaceAllowed:                                            false,
		CoocLearningAllowed:                                            false,
		DeltaHarvestAllowed:                                            false,
		BodyMutationAllowed:                                            false,
		RequiresWriter:                                                 true,
		WriterReady:                                                    false,
		RollbackRequired:                                               true,
		RequiresRollback:                                               true,
		RollbackReady:                                                  false,
		ReadOnly:                                                       true,
		ReplayOnly:                                                     true,
		SourceSchema:                                                   sourceContract.Schema,
		SourceStatus:                                                   sourceContract.Status,
		SourceTarget:                                                   sourceContract.Target,
		SourceReport:                                                   writerContractPath,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID:        sourceContract.WeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReady:     sourceContract.WeightedAdmissionResonanceGraftAdmissionWriterContractReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID:  sourceContract.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash:      sourceContract.WriterContractHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack:  sourceContract.ReadBackHash,
		SourceWriterContractReceiptShape:                                      sourceContract.ReceiptShape,
		SourceWriterContractKind:                                              sourceContract.WriterContractKind,
		SourceWriterContractMode:                                              sourceContract.WriterContractMode,
		SourceWriterContractStage:                                             sourceContract.WriterContractStage,
		SourceWriterContractContractState:                                     sourceContract.ContractState,
		SourceWriterContractContractAction:                                    sourceContract.ContractAction,
		SourceWriterContractWriterAction:                                      sourceContract.WriterAction,
		SourceWriterContractRollbackAction:                                    sourceContract.RollbackAction,
		SourceWriterContractWriterContract:                                    sourceContract.WriterContract,
		SourceWriterContractRollbackContract:                                  sourceContract.RollbackContract,
		SourceWriterContractAdmissionLedgerContract:                           sourceContract.AdmissionLedgerContract,
		SourceWriterContractWriterContractShape:                               sourceContract.WriterContractShape,
		SourceWriterContractRollbackContractShape:                             sourceContract.RollbackContractShape,
		SourceWriterContractLedgerContractShape:                               sourceContract.LedgerContractShape,
		SourceWriterContractWriteScope:                                        sourceContract.WriteScope,
		SourceWriterContractRollbackScope:                                     sourceContract.RollbackScope,
		SourceWriterContractLedgerMode:                                        sourceContract.LedgerMode,
		SourceWriterContractWriterContractPresent:                             sourceContract.WriterContractPresent,
		SourceWriterContractRollbackContractPresent:                           sourceContract.RollbackContractPresent,
		SourceWriterContractLedgerContractPresent:                             sourceContract.LedgerContractPresent,
		SourceWriterContractContractsReady:                                    sourceContract.ContractsReady,
		SourceWriterContractBodyTarget:                                        sourceContract.BodyTarget,
		SourceWriterContractReason:                                            sourceContract.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack,
		SourceWriterInventoryReceiptShape:                                     sourceContract.SourceWriterInventoryReceiptShape,
		SourceWriterInventoryKind:                                             sourceContract.SourceWriterInventoryKind,
		SourceWriterInventoryMode:                                             sourceContract.SourceWriterInventoryMode,
		SourceWriterInventoryStage:                                            sourceContract.SourceWriterInventoryStage,
		SourceWriterInventoryWriterState:                                      sourceContract.SourceWriterInventoryWriterState,
		SourceWriterInventoryWriterAction:                                     sourceContract.SourceWriterInventoryWriterAction,
		SourceWriterInventoryRollbackState:                                    sourceContract.SourceWriterInventoryRollbackState,
		SourceWriterInventoryRollbackAction:                                   sourceContract.SourceWriterInventoryRollbackAction,
		SourceWriterInventoryInventoryState:                                   sourceContract.SourceWriterInventoryInventoryState,
		SourceWriterInventoryInventoryAction:                                  sourceContract.SourceWriterInventoryInventoryAction,
		SourceWriterInventoryWriterContract:                                   sourceContract.SourceWriterInventoryWriterContract,
		SourceWriterInventoryRollbackContract:                                 sourceContract.SourceWriterInventoryRollbackContract,
		SourceWriterInventoryAdmissionLedgerContract:                          sourceContract.SourceWriterInventoryAdmissionLedgerContract,
		SourceWriterInventoryWriterContractPresent:                            sourceContract.SourceWriterInventoryWriterContractPresent,
		SourceWriterInventoryRollbackContractPresent:                          sourceContract.SourceWriterInventoryRollbackContractPresent,
		SourceWriterInventoryLedgerContractPresent:                            sourceContract.SourceWriterInventoryLedgerContractPresent,
		SourceWriterInventoryContractsReady:                                   sourceContract.SourceWriterInventoryContractsReady,
		SourceWriterInventoryBodyTarget:                                       sourceContract.SourceWriterInventoryBodyTarget,
		SourceWriterInventoryReason:                                           sourceContract.SourceWriterInventoryReason,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash,
		SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack,
		SourceWriterPreflightReceiptShape:                                     sourceContract.SourceWriterPreflightReceiptShape,
		SourceWriterPreflightKind:                                             sourceContract.SourceWriterPreflightKind,
		SourceWriterPreflightMode:                                             sourceContract.SourceWriterPreflightMode,
		SourceWriterPreflightStage:                                            sourceContract.SourceWriterPreflightStage,
		SourceWriterPreflightWriterState:                                      sourceContract.SourceWriterPreflightWriterState,
		SourceWriterPreflightWriterAction:                                     sourceContract.SourceWriterPreflightWriterAction,
		SourceWriterPreflightRollbackState:                                    sourceContract.SourceWriterPreflightRollbackState,
		SourceWriterPreflightRollbackAction:                                   sourceContract.SourceWriterPreflightRollbackAction,
		SourceWriterPreflightAdmissionRequired:                                sourceContract.SourceWriterPreflightAdmissionRequired,
		SourceWriterPreflightShadowOnly:                                       sourceContract.SourceWriterPreflightShadowOnly,
		SourceWriterPreflightGraftAllowed:                                     sourceContract.SourceWriterPreflightGraftAllowed,
		SourceWriterPreflightDryRunOnly:                                       sourceContract.SourceWriterPreflightDryRunOnly,
		SourceWriterPreflightLiveReady:                                        sourceContract.SourceWriterPreflightLiveReady,
		SourceWriterPreflightRawDreamTextAllowed:                              sourceContract.SourceWriterPreflightRawDreamTextAllowed,
		SourceWriterPreflightRawDreamTextObserved:                             sourceContract.SourceWriterPreflightRawDreamTextObserved,
		SourceWriterPreflightRawDreamTextForwarded:                            sourceContract.SourceWriterPreflightRawDreamTextForwarded,
		SourceWriterPreflightJanusSurfaceAllowed:                              sourceContract.SourceWriterPreflightJanusSurfaceAllowed,
		SourceWriterPreflightCoocLearningAllowed:                              sourceContract.SourceWriterPreflightCoocLearningAllowed,
		SourceWriterPreflightDeltaHarvestAllowed:                              sourceContract.SourceWriterPreflightDeltaHarvestAllowed,
		SourceWriterPreflightBodyMutationAllowed:                              sourceContract.SourceWriterPreflightBodyMutationAllowed,
		SourceWriterPreflightRequiresWriter:                                   sourceContract.SourceWriterPreflightRequiresWriter,
		SourceWriterPreflightWriterReady:                                      sourceContract.SourceWriterPreflightWriterReady,
		SourceWriterPreflightRollbackRequired:                                 sourceContract.SourceWriterPreflightRollbackRequired,
		SourceWriterPreflightRequiresRollback:                                 sourceContract.SourceWriterPreflightRequiresRollback,
		SourceWriterPreflightRollbackReady:                                    sourceContract.SourceWriterPreflightRollbackReady,
		SourceWriterPreflightReadOnly:                                         sourceContract.SourceWriterPreflightReadOnly,
		SourceWriterPreflightReplayOnly:                                       sourceContract.SourceWriterPreflightReplayOnly,
		SourceWriterPreflightWriteAllowed:                                     sourceContract.SourceWriterPreflightWriteAllowed,
		SourceWriterPreflightAdmissionAllowed:                                 sourceContract.SourceWriterPreflightAdmissionAllowed,
		SourceWriterPreflightLiveAdmissionEnabled:                             sourceContract.SourceWriterPreflightLiveAdmissionEnabled,
		SourceWriterPreflightMutatesState:                                     sourceContract.SourceWriterPreflightMutatesState,
		SourceWriterPreflightBodyTarget:                                       sourceContract.SourceWriterPreflightBodyTarget,
		SourceWriterPreflightPassed:                                           sourceContract.SourceWriterPreflightPassed,
		SourceWriterPreflightReason:                                           sourceContract.SourceWriterPreflightReason,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID:             sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady:          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash:           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceStageState:                                                      sourceContract.SourceStageState,
		SourceStageAction:                                                     sourceContract.SourceStageAction,
		SourceLiveStageReceiptShape:                                           sourceContract.SourceLiveStageReceiptShape,
		SourceLiveStageKind:                                                   sourceContract.SourceLiveStageKind,
		SourceLiveStageMode:                                                   sourceContract.SourceLiveStageMode,
		SourceLiveStageStage:                                                  sourceContract.SourceLiveStageStage,
		SourceLiveStageAdmissionRequired:                                      sourceContract.SourceLiveStageAdmissionRequired,
		SourceLiveStageShadowOnly:                                             sourceContract.SourceLiveStageShadowOnly,
		SourceLiveStageGraftAllowed:                                           sourceContract.SourceLiveStageGraftAllowed,
		SourceLiveStageDryRunOnly:                                             sourceContract.SourceLiveStageDryRunOnly,
		SourceLiveStageLiveReady:                                              sourceContract.SourceLiveStageLiveReady,
		SourceLiveStageRawDreamTextAllowed:                                    sourceContract.SourceLiveStageRawDreamTextAllowed,
		SourceLiveStageRawDreamTextObserved:                                   sourceContract.SourceLiveStageRawDreamTextObserved,
		SourceLiveStageRawDreamTextForwarded:                                  sourceContract.SourceLiveStageRawDreamTextForwarded,
		SourceLiveStageJanusSurfaceAllowed:                                    sourceContract.SourceLiveStageJanusSurfaceAllowed,
		SourceLiveStageCoocLearningAllowed:                                    sourceContract.SourceLiveStageCoocLearningAllowed,
		SourceLiveStageDeltaHarvestAllowed:                                    sourceContract.SourceLiveStageDeltaHarvestAllowed,
		SourceLiveStageBodyMutationAllowed:                                    sourceContract.SourceLiveStageBodyMutationAllowed,
		SourceLiveStageRequiresWriter:                                         sourceContract.SourceLiveStageRequiresWriter,
		SourceLiveStageWriterReady:                                            sourceContract.SourceLiveStageWriterReady,
		SourceLiveStageRollbackRequired:                                       sourceContract.SourceLiveStageRollbackRequired,
		SourceLiveStageRequiresRollback:                                       sourceContract.SourceLiveStageRequiresRollback,
		SourceLiveStageRollbackReady:                                          sourceContract.SourceLiveStageRollbackReady,
		SourceLiveStageReadOnly:                                               sourceContract.SourceLiveStageReadOnly,
		SourceLiveStageReplayOnly:                                             sourceContract.SourceLiveStageReplayOnly,
		SourceLiveStageWriteAllowed:                                           sourceContract.SourceLiveStageWriteAllowed,
		SourceLiveStageAdmissionAllowed:                                       sourceContract.SourceLiveStageAdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:                                   sourceContract.SourceLiveStageLiveAdmissionEnabled,
		SourceLiveStageMutatesState:                                           sourceContract.SourceLiveStageMutatesState,
		SourceLiveStageBodyTarget:                                             sourceContract.SourceLiveStageBodyTarget,
		SourceLiveStagePassed:                                                 sourceContract.SourceLiveStagePassed,
		SourceLiveStageReason:                                                 sourceContract.SourceLiveStageReason,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID:            sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady:         sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID:      sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash:          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack:      sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceEnableState:                                                     sourceContract.SourceEnableState,
		SourceEnableAction:                                                    sourceContract.SourceEnableAction,
		SourceEnableGateReceiptShape:                                          sourceContract.SourceEnableGateReceiptShape,
		SourceEnableGateKind:                                                  sourceContract.SourceEnableGateKind,
		SourceEnableGateMode:                                                  sourceContract.SourceEnableGateMode,
		SourceEnableGateStage:                                                 sourceContract.SourceEnableGateStage,
		SourceEnableGateAdmissionRequired:                                     sourceContract.SourceEnableGateAdmissionRequired,
		SourceEnableGateShadowOnly:                                            sourceContract.SourceEnableGateShadowOnly,
		SourceEnableGateGraftAllowed:                                          sourceContract.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:                                            sourceContract.SourceEnableGateDryRunOnly,
		SourceEnableGateLiveReady:                                             sourceContract.SourceEnableGateLiveReady,
		SourceEnableGateRawDreamTextAllowed:                                   sourceContract.SourceEnableGateRawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:                                  sourceContract.SourceEnableGateRawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded:                                 sourceContract.SourceEnableGateRawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:                                   sourceContract.SourceEnableGateJanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:                                   sourceContract.SourceEnableGateCoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:                                   sourceContract.SourceEnableGateDeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:                                   sourceContract.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:                                      sourceContract.SourceEnableGateRollbackRequired,
		SourceEnableGateReadOnly:                                              sourceContract.SourceEnableGateReadOnly,
		SourceEnableGateReplayOnly:                                            sourceContract.SourceEnableGateReplayOnly,
		SourceEnableGateWriteAllowed:                                          sourceContract.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:                                      sourceContract.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:                                  sourceContract.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:                                          sourceContract.SourceEnableGateMutatesState,
		SourceEnableGateBodyTarget:                                            sourceContract.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                                                sourceContract.SourceEnableGatePassed,
		SourceEnableGateReason:                                                sourceContract.SourceEnableGateReason,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:                sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:             sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID:          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack:          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourceSwitchState:                                                     sourceContract.SourceSwitchState,
		SourceSwitchAction:                                                    sourceContract.SourceSwitchAction,
		SourceSwitchReceiptShape:                                              sourceContract.SourceSwitchReceiptShape,
		SourceSwitchKind:                                                      sourceContract.SourceSwitchKind,
		SourceSwitchMode:                                                      sourceContract.SourceSwitchMode,
		SourceSwitchStage:                                                     sourceContract.SourceSwitchStage,
		SourceSwitchAdmissionRequired:                                         sourceContract.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:                                                sourceContract.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:                                              sourceContract.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:                                                sourceContract.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:                                                 sourceContract.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:                                       sourceContract.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                                      sourceContract.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                                     sourceContract.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                                       sourceContract.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                                       sourceContract.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                                       sourceContract.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                                       sourceContract.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:                                          sourceContract.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:                                                  sourceContract.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:                                                sourceContract.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:                                              sourceContract.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:                                          sourceContract.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                                      sourceContract.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:                                              sourceContract.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:                                                sourceContract.SourceSwitchBodyTarget,
		SourceSwitchPassed:                                                    sourceContract.SourceSwitchPassed,
		SourceSwitchReason:                                                    sourceContract.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:             sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                       sourceContract.SourcePromotion,
		SourcePromotionAction:                                                 sourceContract.SourcePromotionAction,
		SourcePromotionReceiptShape:                                           sourceContract.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                   sourceContract.SourcePromotionKind,
		SourcePromotionMode:                                                   sourceContract.SourcePromotionMode,
		SourcePromotionStage:                                                  sourceContract.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                      sourceContract.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                             sourceContract.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                           sourceContract.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                             sourceContract.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                              sourceContract.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                    sourceContract.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                   sourceContract.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                  sourceContract.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                    sourceContract.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                    sourceContract.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                    sourceContract.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                    sourceContract.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                       sourceContract.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                               sourceContract.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                             sourceContract.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                           sourceContract.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                       sourceContract.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                   sourceContract.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                           sourceContract.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                             sourceContract.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                 sourceContract.SourcePromotionPassed,
		SourcePromotionReason:                                                 sourceContract.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady:  sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                 sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:           sourceContract.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:        sourceContract.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                 sourceContract.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:              sourceContract.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                      sourceContract.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                   sourceContract.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                           sourceContract.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                        sourceContract.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                      sourceContract.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                   sourceContract.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                       sourceContract.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                    sourceContract.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                         sourceContract.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                      sourceContract.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                            sourceContract.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                         sourceContract.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                           sourceContract.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                 sourceContract.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                      sourceContract.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                    sourceContract.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                              sourceContract.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                              sourceContract.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                     sourceContract.BodySmokeWeighted,
		NanoDirectRunner:                                                      sourceContract.NanoDirectRunner,
		NanoDirectFinalGate:                                                   sourceContract.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                          sourceContract.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                               sourceContract.BoundaryReportFullChain,
		SourceAuthorityGranted:                                                sourceContract.SourceAuthorityGranted,
		AuthorityGranted:                                                      false,
		ContractsReady:                                                        false,
		WriteAllowed:                                                          false,
		AdmissionAllowed:                                                      false,
		LiveAdmissionEnabled:                                                  false,
		MutatesState:                                                          false,
		BodyTarget:                                                            "none",
		Passed:                                                                true,
		Reason:                                                                "weighted resonance shadow graft admission ledger blocked by blocked writer contract; ledger receipt append remains closed",
	}
	inventory.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerCausalID(inventory)
	inventory.AdmissionLedgerHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerHash(inventory)
	inventory.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReadBackHash(inventory)
	inventory.WeightedAdmissionResonanceGraftAdmissionLedgerID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerID(inventory)
	if inventory.CausalID == "" ||
		inventory.AdmissionLedgerHash == "" ||
		inventory.ReadBackHash == "" ||
		inventory.WeightedAdmissionResonanceGraftAdmissionLedgerID == "" ||
		inventory.AdmissionLedgerHash == inventory.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger read-back proof failed")
	}
	raw, err := json.MarshalIndent(inventory, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-ledger] pass: resonance_graft_admission_ledger_report=%s resonance_graft_admission_writer_contract_report=%s\n", outputPath, writerContractPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission ledger schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema)
	}
	if report.Status != "shadow_graft_admission_ledger_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger status mismatch: got %q want %q", report.Status, "shadow_graft_admission_ledger_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission ledger target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger" {
		return fmt.Errorf("weighted admission resonance graft admission ledger target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_ledger")
	}
	if report.TargetMode != "closed_admission_ledger_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger target_mode mismatch: got %q want %q", report.TargetMode, "closed_admission_ledger_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_writer_contract_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission ledger writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_writer_contract")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission ledger rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_writer_contract")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission ledger stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission ledger enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission ledger enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission ledger switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission ledger switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission ledger promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.InventoryState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger inventory_state mismatch: got %q want %q", report.InventoryState, "blocked")
	}
	if report.InventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission ledger inventory_action mismatch: got %q want %q", report.InventoryAction, "reject_blocked_writer_preflight")
	}
	if report.ContractState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger contract_state mismatch: got %q want %q", report.ContractState, "blocked")
	}
	if report.ContractAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission ledger contract_action mismatch: got %q want %q", report.ContractAction, "reject_blocked_writer_inventory")
	}
	if report.WriterContract != "none" || report.RollbackContract != "none" || report.AdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger contracts unexpectedly named")
	}
	if report.WriterContractShape != "none" || report.RollbackContractShape != "none" || report.LedgerContractShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger contract shapes unexpectedly named")
	}
	if report.WriteScope != "none" || report.RollbackScope != "none" || report.LedgerMode != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger scopes unexpectedly opened")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission ledger ledger state/action mismatch")
	}
	if report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger append shape unexpectedly opened")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission ledger receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_ledger_receipt")
	}
	if report.AdmissionLedgerKind != "shadow_graft_admission_ledger" ||
		report.AdmissionLedgerMode != "closed_writer_contract_ledger_guard" ||
		report.AdmissionLedgerStage != "pre_ledger_append_graft_admission_ledger" {
		return fmt.Errorf("weighted admission resonance graft admission ledger shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_ledger_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerReady},
		{"weighted_admission_resonance_graft_admission_writer_contract_consumed", report.WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed},
		{"weighted_admission_resonance_graft_admission_writer_contract_required", report.WeightedAdmissionResonanceGraftAdmissionWriterContractRequired},
		{"next_step_blocked_without_resonance_graft_admission_ledger", report.NextStepBlockedWithoutResonanceGraftAdmissionLedger},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReady},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger %s not ready", required.name)
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
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"source_writer_contract_writer_contract_present", report.SourceWriterContractWriterContractPresent},
		{"source_writer_contract_rollback_contract_present", report.SourceWriterContractRollbackContractPresent},
		{"source_writer_contract_ledger_contract_present", report.SourceWriterContractLedgerContractPresent},
		{"source_writer_contract_contracts_ready", report.SourceWriterContractContractsReady},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_ledger_id", report.WeightedAdmissionResonanceGraftAdmissionLedgerID},
		{"causal_id", report.CausalID},
		{"admission_ledger_hash", report.AdmissionLedgerHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack},
		{"source_writer_contract_reason", report.SourceWriterContractReason},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_writer_contract_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_writer_contract_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceWriterContractReceiptShape != "weighted_resonance_shadow_graft_admission_writer_contract_receipt" ||
		report.SourceWriterContractKind != "shadow_graft_admission_writer_contract" ||
		report.SourceWriterContractMode != "closed_writer_inventory_contract_guard" ||
		report.SourceWriterContractStage != "pre_admission_ledger_graft_admission_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer contract shape mismatch")
	}
	if report.SourceWriterContractContractState != "blocked" ||
		report.SourceWriterContractContractAction != "reject_blocked_writer_inventory" ||
		report.SourceWriterContractWriterAction != "reject_blocked_writer_inventory" ||
		report.SourceWriterContractRollbackAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer contract state mismatch")
	}
	if report.SourceWriterContractWriterContract != "none" ||
		report.SourceWriterContractRollbackContract != "none" ||
		report.SourceWriterContractAdmissionLedgerContract != "none" ||
		report.SourceWriterContractWriterContractShape != "none" ||
		report.SourceWriterContractRollbackContractShape != "none" ||
		report.SourceWriterContractLedgerContractShape != "none" ||
		report.SourceWriterContractWriteScope != "none" ||
		report.SourceWriterContractRollbackScope != "none" ||
		report.SourceWriterContractLedgerMode != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer contract unexpectedly opened")
	}
	if report.SourceWriterContractBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_contract_body_target mismatch: got %q want %q", report.SourceWriterContractBodyTarget, "none")
	}
	if report.SourceWriterContractReason != "weighted resonance shadow graft admission writer contract blocked by blocked writer inventory; writer, rollback, and ledger contract shapes remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_contract_reason mismatch: got %q", report.SourceWriterContractReason)
	}
	if report.SourceWriterInventoryReceiptShape != "weighted_resonance_shadow_graft_admission_writer_inventory_receipt" ||
		report.SourceWriterInventoryKind != "shadow_graft_admission_writer_inventory" ||
		report.SourceWriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		report.SourceWriterInventoryStage != "pre_writer_contract_graft_admission_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer inventory shape mismatch")
	}
	if report.SourceWriterInventoryWriterState != "blocked" ||
		report.SourceWriterInventoryWriterAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryRollbackState != "blocked" ||
		report.SourceWriterInventoryRollbackAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryInventoryState != "blocked" ||
		report.SourceWriterInventoryInventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer inventory state mismatch")
	}
	if report.SourceWriterInventoryWriterContract != "none" ||
		report.SourceWriterInventoryRollbackContract != "none" ||
		report.SourceWriterInventoryAdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer inventory contracts unexpectedly named")
	}
	if report.SourceWriterInventoryBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_inventory_body_target mismatch: got %q want %q", report.SourceWriterInventoryBodyTarget, "none")
	}
	if report.SourceWriterInventoryReason != "weighted resonance shadow graft admission writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_inventory_reason mismatch: got %q", report.SourceWriterInventoryReason)
	}
	if report.SourceWriterPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_writer_preflight_receipt" ||
		report.SourceWriterPreflightKind != "shadow_graft_admission_writer_preflight" ||
		report.SourceWriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.SourceWriterPreflightStage != "pre_writer_inventory_graft_admission_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer preflight shape mismatch")
	}
	if report.SourceWriterPreflightWriterState != "blocked" ||
		report.SourceWriterPreflightWriterAction != "reject_blocked_live_stage" ||
		report.SourceWriterPreflightRollbackState != "blocked" ||
		report.SourceWriterPreflightRollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer preflight state mismatch")
	}
	if report.SourceWriterPreflightBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_preflight_body_target mismatch: got %q want %q", report.SourceWriterPreflightBodyTarget, "none")
	}
	if report.SourceWriterPreflightReason != "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_writer_preflight_reason mismatch: got %q", report.SourceWriterPreflightReason)
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_live_stage" ||
		report.SourceLiveStageMode != "closed_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "pre_writer_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission ledger source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		report.SourceEnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.SourceEnableGateMode != "closed_switch_enable_guard" ||
		report.SourceEnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission ledger source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission ledger source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionLedgerID, "weighted-resonance-graft-admission-ledger-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-ledger-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger causal prefix mismatch")
	}
	if !strings.HasPrefix(report.AdmissionLedgerHash, "weighted-resonance-graft-admission-ledger-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-ledger-read-") ||
		report.AdmissionLedgerHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID, "weighted-resonance-graft-admission-writer-inventory-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryCausalID, "weighted-resonance-graft-admission-writer-inventory-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryHash, "weighted-resonance-graft-admission-writer-inventory-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryReadBack, "weighted-resonance-graft-admission-writer-inventory-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer inventory mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID, "weighted-resonance-graft-admission-writer-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID, "weighted-resonance-graft-admission-writer-preflight-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash, "weighted-resonance-graft-admission-writer-preflight-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBack, "weighted-resonance-graft-admission-writer-preflight-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer preflight mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID, "weighted-resonance-graft-admission-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID, "weighted-resonance-graft-admission-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash, "weighted-resonance-graft-admission-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack, "weighted-resonance-graft-admission-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID, "weighted-resonance-graft-admission-enable-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash, "weighted-resonance-graft-admission-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack, "weighted-resonance-graft-admission-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger causal_id mismatch")
	}
	if report.AdmissionLedgerHash == "" || report.AdmissionLedgerHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger admission_ledger_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger read_back_hash mismatch")
	}
	if report.AdmissionLedgerHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionLedgerID == "" || report.WeightedAdmissionResonanceGraftAdmissionLedgerID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger id mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID, "weighted-resonance-graft-admission-writer-contract-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractCausalID, "weighted-resonance-graft-admission-writer-contract-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash, "weighted-resonance-graft-admission-writer-contract-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack, "weighted-resonance-graft-admission-writer-contract-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger source writer contract mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission ledger blocked by blocked writer contract; ledger receipt append remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport) string {
	h := hashJSON(struct {
		SourceWriterContractID   string `json:"source_writer_contract_id"`
		SourceWriterContractRead string `json:"source_writer_contract_read_back_hash"`
		SourceWriterInventoryID  string `json:"source_writer_inventory_id"`
		Target                   string `json:"target"`
		AdmissionLedgerKind      string `json:"admission_ledger_kind"`
		AdmissionLedgerStage     string `json:"admission_ledger_stage"`
	}{
		SourceWriterContractID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWriterContractRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack,
		SourceWriterInventoryID:  sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID,
		Target:                   sw.Target,
		AdmissionLedgerKind:      sw.AdmissionLedgerKind,
		AdmissionLedgerStage:     sw.AdmissionLedgerStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport) string {
	h := hashJSON(struct {
		CausalID                  string `json:"causal_id"`
		SourceWriterContractID    string `json:"source_writer_contract_id"`
		SourceWriterContractHash  string `json:"source_writer_contract_hash"`
		SourceWriterContractRead  string `json:"source_writer_contract_read_back_hash"`
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
		LedgerState               string `json:"ledger_state"`
		LedgerAction              string `json:"ledger_action"`
		LedgerContract            string `json:"ledger_contract"`
		LedgerEntrypoint          string `json:"ledger_entrypoint"`
		LedgerReceiptShape        string `json:"ledger_receipt_shape"`
		LedgerWriteScope          string `json:"ledger_write_scope"`
		LedgerReady               bool   `json:"ledger_ready"`
		LedgerAppendAllowed       bool   `json:"ledger_append_allowed"`
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
		AdmissionLedgerMode       string `json:"admission_ledger_mode"`
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
		SourceWriterContractID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWriterContractHash:  sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash,
		SourceWriterContractRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack,
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
		LedgerState:               sw.LedgerState,
		LedgerAction:              sw.LedgerAction,
		LedgerContract:            sw.LedgerContract,
		LedgerEntrypoint:          sw.LedgerEntrypoint,
		LedgerReceiptShape:        sw.LedgerReceiptShape,
		LedgerWriteScope:          sw.LedgerWriteScope,
		LedgerReady:               sw.LedgerReady,
		LedgerAppendAllowed:       sw.LedgerAppendAllowed,
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
		AdmissionLedgerMode:       sw.AdmissionLedgerMode,
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
	return "weighted-resonance-graft-admission-ledger-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport) string {
	h := hashJSON(struct {
		AdmissionLedgerHash      string `json:"admission_ledger_hash"`
		SourceWriterContractID   string `json:"source_writer_contract_id"`
		SourceWriterContractRead string `json:"source_writer_contract_read_back_hash"`
		AdmissionLedgerKind      string `json:"admission_ledger_kind"`
		AdmissionLedgerReady     bool   `json:"admission_ledger_ready"`
		WriterContractConsumed   bool   `json:"writer_contract_consumed"`
		LedgerAppendAllowed      bool   `json:"ledger_append_allowed"`
		RequiresWriter           bool   `json:"requires_writer"`
		WriterReady              bool   `json:"writer_ready"`
		RequiresRollback         bool   `json:"requires_rollback"`
		RollbackReady            bool   `json:"rollback_ready"`
		ContractsReady           bool   `json:"contracts_ready"`
		LiveReady                bool   `json:"live_ready"`
		BodyMutation             bool   `json:"body_mutation"`
		LiveAdmission            bool   `json:"live_admission"`
		WriteAllowed             bool   `json:"write_allowed"`
		AdmissionAllowed         bool   `json:"admission_allowed"`
	}{
		AdmissionLedgerHash:      sw.AdmissionLedgerHash,
		SourceWriterContractID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWriterContractRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack,
		AdmissionLedgerKind:      sw.AdmissionLedgerKind,
		AdmissionLedgerReady:     sw.WeightedAdmissionResonanceGraftAdmissionLedgerReady,
		WriterContractConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed,
		LedgerAppendAllowed:      sw.LedgerAppendAllowed,
		RequiresWriter:           sw.RequiresWriter,
		WriterReady:              sw.WriterReady,
		RequiresRollback:         sw.RequiresRollback,
		RollbackReady:            sw.RollbackReady,
		ContractsReady:           sw.ContractsReady,
		LiveReady:                sw.LiveReady,
		BodyMutation:             sw.BodyMutationAllowed,
		LiveAdmission:            sw.LiveAdmissionEnabled,
		WriteAllowed:             sw.WriteAllowed,
		AdmissionAllowed:         sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport) string {
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
		LedgerState               string `json:"ledger_state"`
		LedgerAction              string `json:"ledger_action"`
		LedgerContract            string `json:"ledger_contract"`
		LedgerEntrypoint          string `json:"ledger_entrypoint"`
		LedgerReceiptShape        string `json:"ledger_receipt_shape"`
		LedgerWriteScope          string `json:"ledger_write_scope"`
		LedgerReady               bool   `json:"ledger_ready"`
		LedgerAppendAllowed       bool   `json:"ledger_append_allowed"`
		StageState                string `json:"stage_state"`
		StageAction               string `json:"stage_action"`
		EnableState               string `json:"enable_state"`
		EnableAction              string `json:"enable_action"`
		SwitchState               string `json:"switch_state"`
		SwitchAction              string `json:"switch_action"`
		Promotion                 string `json:"promotion"`
		SourceReport              string `json:"source_report"`
		SourceWriterContractID    string `json:"source_writer_contract_id"`
		SourceWriterContractHash  string `json:"source_writer_contract_hash"`
		SourceWriterContractRead  string `json:"source_writer_contract_read_back_hash"`
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
		AdmissionLedgerHash       string `json:"admission_ledger_hash"`
		ReadBackHash              string `json:"read_back_hash"`
		Ready                     bool   `json:"ready"`
		ReceiptShape              string `json:"receipt_shape"`
		AdmissionLedgerKind       string `json:"admission_ledger_kind"`
		AdmissionLedgerMode       string `json:"admission_ledger_mode"`
		AdmissionLedgerStage      string `json:"admission_ledger_stage"`
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
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger"`
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
		LedgerState:               sw.LedgerState,
		LedgerAction:              sw.LedgerAction,
		LedgerContract:            sw.LedgerContract,
		LedgerEntrypoint:          sw.LedgerEntrypoint,
		LedgerReceiptShape:        sw.LedgerReceiptShape,
		LedgerWriteScope:          sw.LedgerWriteScope,
		LedgerReady:               sw.LedgerReady,
		LedgerAppendAllowed:       sw.LedgerAppendAllowed,
		StageState:                sw.StageState,
		StageAction:               sw.StageAction,
		EnableState:               sw.EnableState,
		EnableAction:              sw.EnableAction,
		SwitchState:               sw.SwitchState,
		SwitchAction:              sw.SwitchAction,
		Promotion:                 sw.Promotion,
		SourceReport:              sw.SourceReport,
		SourceWriterContractID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWriterContractHash:  sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash,
		SourceWriterContractRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReadBack,
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
		AdmissionLedgerHash:       sw.AdmissionLedgerHash,
		ReadBackHash:              sw.ReadBackHash,
		Ready:                     sw.WeightedAdmissionResonanceGraftAdmissionLedgerReady,
		ReceiptShape:              sw.ReceiptShape,
		AdmissionLedgerKind:       sw.AdmissionLedgerKind,
		AdmissionLedgerMode:       sw.AdmissionLedgerMode,
		AdmissionLedgerStage:      sw.AdmissionLedgerStage,
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
		NextStepBlockedWithout:    sw.NextStepBlockedWithoutResonanceGraftAdmissionLedger,
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
	return "weighted-resonance-graft-admission-ledger-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger decode failed: %w", err)
	}
	return report, root, nil
}
