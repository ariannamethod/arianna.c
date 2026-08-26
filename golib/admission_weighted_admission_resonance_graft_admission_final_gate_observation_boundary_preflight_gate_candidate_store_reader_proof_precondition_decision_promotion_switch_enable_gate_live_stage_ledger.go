package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport struct {
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
	ContractState                                                                                                                                                                                 string `json:"contract_state"`
	ContractAction                                                                                                                                                                                string `json:"contract_action"`
	WriterContract                                                                                                                                                                                string `json:"writer_contract"`
	RollbackContract                                                                                                                                                                              string `json:"rollback_contract"`
	AdmissionLedgerContract                                                                                                                                                                       string `json:"admission_ledger_contract"`
	WriterContractShape                                                                                                                                                                           string `json:"writer_contract_shape"`
	RollbackContractShape                                                                                                                                                                         string `json:"rollback_contract_shape"`
	LedgerContractShape                                                                                                                                                                           string `json:"ledger_contract_shape"`
	WriteScope                                                                                                                                                                                    string `json:"write_scope"`
	RollbackScope                                                                                                                                                                                 string `json:"rollback_scope"`
	LedgerMode                                                                                                                                                                                    string `json:"ledger_mode"`
	WriterContractPresent                                                                                                                                                                         bool   `json:"writer_contract_present"`
	RollbackContractPresent                                                                                                                                                                       bool   `json:"rollback_contract_present"`
	LedgerContractPresent                                                                                                                                                                         bool   `json:"ledger_contract_present"`
	LedgerState                                                                                                                                                                                   string `json:"ledger_state"`
	LedgerAction                                                                                                                                                                                  string `json:"ledger_action"`
	LedgerContract                                                                                                                                                                                string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                                                              string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                                                            string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                                                              string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                                                   bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                                                           bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady                   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractConsumed        bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractRequired        bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger                   bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID                      string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_id"`
	ReceiptShape                                                                                                                                                                                  string `json:"receipt_shape"`
	AdmissionLedgerKind                                                                                                                                                                           string `json:"admission_ledger_kind"`
	AdmissionLedgerMode                                                                                                                                                                           string `json:"admission_ledger_mode"`
	AdmissionLedgerStage                                                                                                                                                                          string `json:"admission_ledger_stage"`
	CausalID                                                                                                                                                                                      string `json:"causal_id"`
	AdmissionLedgerHash                                                                                                                                                                           string `json:"admission_ledger_hash"`
	ReadBackHash                                                                                                                                                                                  string `json:"read_back_hash"`
	WriterInventoryVerified                                                                                                                                                                       bool   `json:"writer_inventory_verified"`
	WriterInventoryHashVerified                                                                                                                                                                   bool   `json:"writer_inventory_hash_verified"`
	WriterInventoryReadBackVerified                                                                                                                                                               bool   `json:"writer_inventory_read_back_verified"`
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
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReady     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractCausalID  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_read_back_hash"`
	SourceWriterContractReceiptShape                                                                                                                                                              string `json:"source_writer_contract_receipt_shape"`
	SourceWriterContractKind                                                                                                                                                                      string `json:"source_writer_contract_kind"`
	SourceWriterContractMode                                                                                                                                                                      string `json:"source_writer_contract_mode"`
	SourceWriterContractStage                                                                                                                                                                     string `json:"source_writer_contract_stage"`
	SourceWriterContractContractState                                                                                                                                                             string `json:"source_writer_contract_contract_state"`
	SourceWriterContractContractAction                                                                                                                                                            string `json:"source_writer_contract_contract_action"`
	SourceWriterContractWriterAction                                                                                                                                                              string `json:"source_writer_contract_writer_action"`
	SourceWriterContractRollbackAction                                                                                                                                                            string `json:"source_writer_contract_rollback_action"`
	SourceWriterContractWriterContract                                                                                                                                                            string `json:"source_writer_contract_writer_contract"`
	SourceWriterContractRollbackContract                                                                                                                                                          string `json:"source_writer_contract_rollback_contract"`
	SourceWriterContractAdmissionLedgerContract                                                                                                                                                   string `json:"source_writer_contract_admission_ledger_contract"`
	SourceWriterContractWriterContractShape                                                                                                                                                       string `json:"source_writer_contract_writer_contract_shape"`
	SourceWriterContractRollbackContractShape                                                                                                                                                     string `json:"source_writer_contract_rollback_contract_shape"`
	SourceWriterContractLedgerContractShape                                                                                                                                                       string `json:"source_writer_contract_ledger_contract_shape"`
	SourceWriterContractWriteScope                                                                                                                                                                string `json:"source_writer_contract_write_scope"`
	SourceWriterContractRollbackScope                                                                                                                                                             string `json:"source_writer_contract_rollback_scope"`
	SourceWriterContractLedgerMode                                                                                                                                                                string `json:"source_writer_contract_ledger_mode"`
	SourceWriterContractWriterContractPresent                                                                                                                                                     bool   `json:"source_writer_contract_writer_contract_present"`
	SourceWriterContractRollbackContractPresent                                                                                                                                                   bool   `json:"source_writer_contract_rollback_contract_present"`
	SourceWriterContractLedgerContractPresent                                                                                                                                                     bool   `json:"source_writer_contract_ledger_contract_present"`
	SourceWriterContractContractsReady                                                                                                                                                            bool   `json:"source_writer_contract_contracts_ready"`
	SourceWriterContractBodyTarget                                                                                                                                                                string `json:"source_writer_contract_body_target"`
	SourceWriterContractReason                                                                                                                                                                    string `json:"source_writer_contract_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_read_back_hash"`
	SourceWriterInventoryReceiptShape                                                                                                                                                             string `json:"source_writer_inventory_receipt_shape"`
	SourceWriterInventoryKind                                                                                                                                                                     string `json:"source_writer_inventory_kind"`
	SourceWriterInventoryMode                                                                                                                                                                     string `json:"source_writer_inventory_mode"`
	SourceWriterInventoryStage                                                                                                                                                                    string `json:"source_writer_inventory_stage"`
	SourceWriterInventoryWriterState                                                                                                                                                              string `json:"source_writer_inventory_writer_state"`
	SourceWriterInventoryWriterAction                                                                                                                                                             string `json:"source_writer_inventory_writer_action"`
	SourceWriterInventoryRollbackState                                                                                                                                                            string `json:"source_writer_inventory_rollback_state"`
	SourceWriterInventoryRollbackAction                                                                                                                                                           string `json:"source_writer_inventory_rollback_action"`
	SourceWriterInventoryInventoryState                                                                                                                                                           string `json:"source_writer_inventory_inventory_state"`
	SourceWriterInventoryInventoryAction                                                                                                                                                          string `json:"source_writer_inventory_inventory_action"`
	SourceWriterInventoryWriterContract                                                                                                                                                           string `json:"source_writer_inventory_writer_contract"`
	SourceWriterInventoryRollbackContract                                                                                                                                                         string `json:"source_writer_inventory_rollback_contract"`
	SourceWriterInventoryAdmissionLedgerContract                                                                                                                                                  string `json:"source_writer_inventory_admission_ledger_contract"`
	SourceWriterInventoryWriterContractPresent                                                                                                                                                    bool   `json:"source_writer_inventory_writer_contract_present"`
	SourceWriterInventoryRollbackContractPresent                                                                                                                                                  bool   `json:"source_writer_inventory_rollback_contract_present"`
	SourceWriterInventoryLedgerContractPresent                                                                                                                                                    bool   `json:"source_writer_inventory_ledger_contract_present"`
	SourceWriterInventoryContractsReady                                                                                                                                                           bool   `json:"source_writer_inventory_contracts_ready"`
	SourceWriterInventoryBodyTarget                                                                                                                                                               string `json:"source_writer_inventory_body_target"`
	SourceWriterInventoryReason                                                                                                                                                                   string `json:"source_writer_inventory_reason"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_WRITER_CONTRACT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_LIVE_STAGE_LEDGER_REPORT")
	}
	writerContractPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger output path missing")
	}
	sourceContract, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReportForAssert(writerContractPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReportError(sourceContract, root); err != nil {
		return err
	}
	inventory := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport{
		Schema:                  admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema,
		Status:                  "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run",
		Target:                  "live_route_admission_next_step",
		TargetKind:              "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger",
		TargetMode:              "closed_admission_ledger_guard_dry_run",
		Action:                  "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run",
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
		ReceiptShape:            "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_receipt",
		AdmissionLedgerKind:     "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger",
		AdmissionLedgerMode:     "closed_writer_contract_ledger_guard",
		AdmissionLedgerStage:    "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_pre_ledger_append",
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady:            true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger:            true,
		WriterInventoryVerified:         true,
		WriterInventoryHashVerified:     true,
		WriterInventoryReadBackVerified: true,
		WriterPreflightVerified:         true,
		WriterPreflightHashVerified:     true,
		WriterPreflightReadBackVerified: true,
		LiveStageVerified:               sourceContract.LiveStageVerified,
		LiveStageHashVerified:           sourceContract.LiveStageHashVerified,
		LiveStageReadBackVerified:       sourceContract.LiveStageReadBackVerified,
		EnableGateVerified:              sourceContract.EnableGateVerified,
		EnableGateHashVerified:          sourceContract.EnableGateHashVerified,
		EnableGateReadBackVerified:      sourceContract.EnableGateReadBackVerified,
		SwitchVerified:                  sourceContract.SwitchVerified,
		SwitchHashVerified:              sourceContract.SwitchHashVerified,
		SwitchReadBackVerified:          sourceContract.SwitchReadBackVerified,
		PromotionVerified:               sourceContract.PromotionVerified,
		PromotionHashVerified:           sourceContract.PromotionHashVerified,
		PromotionReadBackVerified:       sourceContract.PromotionReadBackVerified,
		DecisionVerified:                sourceContract.DecisionVerified,
		DecisionHashVerified:            sourceContract.DecisionHashVerified,
		DecisionReadBackVerified:        sourceContract.DecisionReadBackVerified,
		ProofPreconditionVerified:       sourceContract.ProofPreconditionVerified,
		PreconditionHashVerified:        sourceContract.PreconditionHashVerified,
		PreconditionReadBackVerified:    sourceContract.PreconditionReadBackVerified,
		ProofVerified:                   sourceContract.ProofVerified,
		ProofHashVerified:               sourceContract.ProofHashVerified,
		ProofReadBackVerified:           sourceContract.ProofReadBackVerified,
		StoreReaderVerified:             sourceContract.StoreReaderVerified,
		StoreVerified:                   sourceContract.StoreVerified,
		CandidateVerified:               sourceContract.CandidateVerified,
		GateVerified:                    sourceContract.GateVerified,
		PreflightVerified:               sourceContract.PreflightVerified,
		BoundaryVerified:                sourceContract.BoundaryVerified,
		ObservationVerified:             sourceContract.ObservationVerified,
		ReceiverVerified:                sourceContract.ReceiverVerified,
		IntentVerified:                  sourceContract.IntentVerified,
		FinalGateVerified:               sourceContract.FinalGateVerified,
		SealVerified:                    sourceContract.SealVerified,
		PermitVerified:                  sourceContract.PermitVerified,
		AuthorityVerified:               sourceContract.AuthorityVerified,
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
		SourceSchema:                    sourceContract.Schema,
		SourceStatus:                    sourceContract.Status,
		SourceTarget:                    sourceContract.Target,
		SourceReport:                    writerContractPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID:       sourceContract.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReady:    sourceContract.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractCausalID: sourceContract.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash:     sourceContract.WriterContractHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack: sourceContract.ReadBackHash,
		SourceWriterContractReceiptShape:            sourceContract.ReceiptShape,
		SourceWriterContractKind:                    sourceContract.WriterContractKind,
		SourceWriterContractMode:                    sourceContract.WriterContractMode,
		SourceWriterContractStage:                   sourceContract.WriterContractStage,
		SourceWriterContractContractState:           sourceContract.ContractState,
		SourceWriterContractContractAction:          sourceContract.ContractAction,
		SourceWriterContractWriterAction:            sourceContract.WriterAction,
		SourceWriterContractRollbackAction:          sourceContract.RollbackAction,
		SourceWriterContractWriterContract:          sourceContract.WriterContract,
		SourceWriterContractRollbackContract:        sourceContract.RollbackContract,
		SourceWriterContractAdmissionLedgerContract: sourceContract.AdmissionLedgerContract,
		SourceWriterContractWriterContractShape:     sourceContract.WriterContractShape,
		SourceWriterContractRollbackContractShape:   sourceContract.RollbackContractShape,
		SourceWriterContractLedgerContractShape:     sourceContract.LedgerContractShape,
		SourceWriterContractWriteScope:              sourceContract.WriteScope,
		SourceWriterContractRollbackScope:           sourceContract.RollbackScope,
		SourceWriterContractLedgerMode:              sourceContract.LedgerMode,
		SourceWriterContractWriterContractPresent:   sourceContract.WriterContractPresent,
		SourceWriterContractRollbackContractPresent: sourceContract.RollbackContractPresent,
		SourceWriterContractLedgerContractPresent:   sourceContract.LedgerContractPresent,
		SourceWriterContractContractsReady:          sourceContract.ContractsReady,
		SourceWriterContractBodyTarget:              sourceContract.BodyTarget,
		SourceWriterContractReason:                  sourceContract.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack,
		SourceWriterInventoryReceiptShape:            sourceContract.SourceWriterInventoryReceiptShape,
		SourceWriterInventoryKind:                    sourceContract.SourceWriterInventoryKind,
		SourceWriterInventoryMode:                    sourceContract.SourceWriterInventoryMode,
		SourceWriterInventoryStage:                   sourceContract.SourceWriterInventoryStage,
		SourceWriterInventoryWriterState:             sourceContract.SourceWriterInventoryWriterState,
		SourceWriterInventoryWriterAction:            sourceContract.SourceWriterInventoryWriterAction,
		SourceWriterInventoryRollbackState:           sourceContract.SourceWriterInventoryRollbackState,
		SourceWriterInventoryRollbackAction:          sourceContract.SourceWriterInventoryRollbackAction,
		SourceWriterInventoryInventoryState:          sourceContract.SourceWriterInventoryInventoryState,
		SourceWriterInventoryInventoryAction:         sourceContract.SourceWriterInventoryInventoryAction,
		SourceWriterInventoryWriterContract:          sourceContract.SourceWriterInventoryWriterContract,
		SourceWriterInventoryRollbackContract:        sourceContract.SourceWriterInventoryRollbackContract,
		SourceWriterInventoryAdmissionLedgerContract: sourceContract.SourceWriterInventoryAdmissionLedgerContract,
		SourceWriterInventoryWriterContractPresent:   sourceContract.SourceWriterInventoryWriterContractPresent,
		SourceWriterInventoryRollbackContractPresent: sourceContract.SourceWriterInventoryRollbackContractPresent,
		SourceWriterInventoryLedgerContractPresent:   sourceContract.SourceWriterInventoryLedgerContractPresent,
		SourceWriterInventoryContractsReady:          sourceContract.SourceWriterInventoryContractsReady,
		SourceWriterInventoryBodyTarget:              sourceContract.SourceWriterInventoryBodyTarget,
		SourceWriterInventoryReason:                  sourceContract.SourceWriterInventoryReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
		SourceWriterPreflightReceiptShape:          sourceContract.SourceWriterPreflightReceiptShape,
		SourceWriterPreflightKind:                  sourceContract.SourceWriterPreflightKind,
		SourceWriterPreflightMode:                  sourceContract.SourceWriterPreflightMode,
		SourceWriterPreflightStage:                 sourceContract.SourceWriterPreflightStage,
		SourceWriterPreflightWriterState:           sourceContract.SourceWriterPreflightWriterState,
		SourceWriterPreflightWriterAction:          sourceContract.SourceWriterPreflightWriterAction,
		SourceWriterPreflightRollbackState:         sourceContract.SourceWriterPreflightRollbackState,
		SourceWriterPreflightRollbackAction:        sourceContract.SourceWriterPreflightRollbackAction,
		SourceWriterPreflightAdmissionRequired:     sourceContract.SourceWriterPreflightAdmissionRequired,
		SourceWriterPreflightShadowOnly:            sourceContract.SourceWriterPreflightShadowOnly,
		SourceWriterPreflightGraftAllowed:          sourceContract.SourceWriterPreflightGraftAllowed,
		SourceWriterPreflightDryRunOnly:            sourceContract.SourceWriterPreflightDryRunOnly,
		SourceWriterPreflightLiveReady:             sourceContract.SourceWriterPreflightLiveReady,
		SourceWriterPreflightRawDreamTextAllowed:   sourceContract.SourceWriterPreflightRawDreamTextAllowed,
		SourceWriterPreflightRawDreamTextObserved:  sourceContract.SourceWriterPreflightRawDreamTextObserved,
		SourceWriterPreflightRawDreamTextForwarded: sourceContract.SourceWriterPreflightRawDreamTextForwarded,
		SourceWriterPreflightJanusSurfaceAllowed:   sourceContract.SourceWriterPreflightJanusSurfaceAllowed,
		SourceWriterPreflightCoocLearningAllowed:   sourceContract.SourceWriterPreflightCoocLearningAllowed,
		SourceWriterPreflightDeltaHarvestAllowed:   sourceContract.SourceWriterPreflightDeltaHarvestAllowed,
		SourceWriterPreflightBodyMutationAllowed:   sourceContract.SourceWriterPreflightBodyMutationAllowed,
		SourceWriterPreflightRequiresWriter:        sourceContract.SourceWriterPreflightRequiresWriter,
		SourceWriterPreflightWriterReady:           sourceContract.SourceWriterPreflightWriterReady,
		SourceWriterPreflightRollbackRequired:      sourceContract.SourceWriterPreflightRollbackRequired,
		SourceWriterPreflightRequiresRollback:      sourceContract.SourceWriterPreflightRequiresRollback,
		SourceWriterPreflightRollbackReady:         sourceContract.SourceWriterPreflightRollbackReady,
		SourceWriterPreflightReadOnly:              sourceContract.SourceWriterPreflightReadOnly,
		SourceWriterPreflightReplayOnly:            sourceContract.SourceWriterPreflightReplayOnly,
		SourceWriterPreflightWriteAllowed:          sourceContract.SourceWriterPreflightWriteAllowed,
		SourceWriterPreflightAdmissionAllowed:      sourceContract.SourceWriterPreflightAdmissionAllowed,
		SourceWriterPreflightLiveAdmissionEnabled:  sourceContract.SourceWriterPreflightLiveAdmissionEnabled,
		SourceWriterPreflightMutatesState:          sourceContract.SourceWriterPreflightMutatesState,
		SourceWriterPreflightBodyTarget:            sourceContract.SourceWriterPreflightBodyTarget,
		SourceWriterPreflightPassed:                sourceContract.SourceWriterPreflightPassed,
		SourceWriterPreflightReason:                sourceContract.SourceWriterPreflightReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack,
		SourceStageState:                     sourceContract.SourceStageState,
		SourceStageAction:                    sourceContract.SourceStageAction,
		SourceLiveStageReceiptShape:          sourceContract.SourceLiveStageReceiptShape,
		SourceLiveStageKind:                  sourceContract.SourceLiveStageKind,
		SourceLiveStageMode:                  sourceContract.SourceLiveStageMode,
		SourceLiveStageStage:                 sourceContract.SourceLiveStageStage,
		SourceLiveStageAdmissionRequired:     sourceContract.SourceLiveStageAdmissionRequired,
		SourceLiveStageShadowOnly:            sourceContract.SourceLiveStageShadowOnly,
		SourceLiveStageGraftAllowed:          sourceContract.SourceLiveStageGraftAllowed,
		SourceLiveStageDryRunOnly:            sourceContract.SourceLiveStageDryRunOnly,
		SourceLiveStageLiveReady:             sourceContract.SourceLiveStageLiveReady,
		SourceLiveStageRawDreamTextAllowed:   sourceContract.SourceLiveStageRawDreamTextAllowed,
		SourceLiveStageRawDreamTextObserved:  sourceContract.SourceLiveStageRawDreamTextObserved,
		SourceLiveStageRawDreamTextForwarded: sourceContract.SourceLiveStageRawDreamTextForwarded,
		SourceLiveStageJanusSurfaceAllowed:   sourceContract.SourceLiveStageJanusSurfaceAllowed,
		SourceLiveStageCoocLearningAllowed:   sourceContract.SourceLiveStageCoocLearningAllowed,
		SourceLiveStageDeltaHarvestAllowed:   sourceContract.SourceLiveStageDeltaHarvestAllowed,
		SourceLiveStageBodyMutationAllowed:   sourceContract.SourceLiveStageBodyMutationAllowed,
		SourceLiveStageRequiresWriter:        sourceContract.SourceLiveStageRequiresWriter,
		SourceLiveStageWriterReady:           sourceContract.SourceLiveStageWriterReady,
		SourceLiveStageRollbackRequired:      sourceContract.SourceLiveStageRollbackRequired,
		SourceLiveStageRequiresRollback:      sourceContract.SourceLiveStageRequiresRollback,
		SourceLiveStageRollbackReady:         sourceContract.SourceLiveStageRollbackReady,
		SourceLiveStageReadOnly:              sourceContract.SourceLiveStageReadOnly,
		SourceLiveStageReplayOnly:            sourceContract.SourceLiveStageReplayOnly,
		SourceLiveStageWriteAllowed:          sourceContract.SourceLiveStageWriteAllowed,
		SourceLiveStageAdmissionAllowed:      sourceContract.SourceLiveStageAdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:  sourceContract.SourceLiveStageLiveAdmissionEnabled,
		SourceLiveStageMutatesState:          sourceContract.SourceLiveStageMutatesState,
		SourceLiveStageBodyTarget:            sourceContract.SourceLiveStageBodyTarget,
		SourceLiveStagePassed:                sourceContract.SourceLiveStagePassed,
		SourceLiveStageReason:                sourceContract.SourceLiveStageReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack,
		SourceEnableState:                     sourceContract.SourceEnableState,
		SourceEnableAction:                    sourceContract.SourceEnableAction,
		SourceEnableGateReceiptShape:          sourceContract.SourceEnableGateReceiptShape,
		SourceEnableGateKind:                  sourceContract.SourceEnableGateKind,
		SourceEnableGateMode:                  sourceContract.SourceEnableGateMode,
		SourceEnableGateStage:                 sourceContract.SourceEnableGateStage,
		SourceEnableGateAdmissionRequired:     sourceContract.SourceEnableGateAdmissionRequired,
		SourceEnableGateShadowOnly:            sourceContract.SourceEnableGateShadowOnly,
		SourceEnableGateGraftAllowed:          sourceContract.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:            sourceContract.SourceEnableGateDryRunOnly,
		SourceEnableGateLiveReady:             sourceContract.SourceEnableGateLiveReady,
		SourceEnableGateRawDreamTextAllowed:   sourceContract.SourceEnableGateRawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:  sourceContract.SourceEnableGateRawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded: sourceContract.SourceEnableGateRawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:   sourceContract.SourceEnableGateJanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:   sourceContract.SourceEnableGateCoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:   sourceContract.SourceEnableGateDeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:   sourceContract.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:      sourceContract.SourceEnableGateRollbackRequired,
		SourceEnableGateReadOnly:              sourceContract.SourceEnableGateReadOnly,
		SourceEnableGateReplayOnly:            sourceContract.SourceEnableGateReplayOnly,
		SourceEnableGateWriteAllowed:          sourceContract.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:      sourceContract.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:  sourceContract.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:          sourceContract.SourceEnableGateMutatesState,
		SourceEnableGateBodyTarget:            sourceContract.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                sourceContract.SourceEnableGatePassed,
		SourceEnableGateReason:                sourceContract.SourceEnableGateReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack,
		SourceSwitchState:                 sourceContract.SourceSwitchState,
		SourceSwitchAction:                sourceContract.SourceSwitchAction,
		SourceSwitchReceiptShape:          sourceContract.SourceSwitchReceiptShape,
		SourceSwitchKind:                  sourceContract.SourceSwitchKind,
		SourceSwitchMode:                  sourceContract.SourceSwitchMode,
		SourceSwitchStage:                 sourceContract.SourceSwitchStage,
		SourceSwitchAdmissionRequired:     sourceContract.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:            sourceContract.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:          sourceContract.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:            sourceContract.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:             sourceContract.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:   sourceContract.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:  sourceContract.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded: sourceContract.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:   sourceContract.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:   sourceContract.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:   sourceContract.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:   sourceContract.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:      sourceContract.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:              sourceContract.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:            sourceContract.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:          sourceContract.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:      sourceContract.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:  sourceContract.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:          sourceContract.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:            sourceContract.SourceSwitchBodyTarget,
		SourceSwitchPassed:                sourceContract.SourceSwitchPassed,
		SourceSwitchReason:                sourceContract.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID:       sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash:     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack,
		SourcePromotion:                      sourceContract.SourcePromotion,
		SourcePromotionAction:                sourceContract.SourcePromotionAction,
		SourcePromotionReceiptShape:          sourceContract.SourcePromotionReceiptShape,
		SourcePromotionKind:                  sourceContract.SourcePromotionKind,
		SourcePromotionMode:                  sourceContract.SourcePromotionMode,
		SourcePromotionStage:                 sourceContract.SourcePromotionStage,
		SourcePromotionAdmissionRequired:     sourceContract.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:            sourceContract.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:          sourceContract.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:            sourceContract.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:             sourceContract.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:   sourceContract.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:  sourceContract.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded: sourceContract.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:   sourceContract.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:   sourceContract.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:   sourceContract.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:   sourceContract.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:      sourceContract.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:              sourceContract.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:            sourceContract.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:          sourceContract.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:      sourceContract.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:  sourceContract.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:          sourceContract.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:            sourceContract.SourcePromotionBodyTarget,
		SourcePromotionPassed:                sourceContract.SourcePromotionPassed,
		SourcePromotionReason:                sourceContract.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:    sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady: sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:            sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady:         sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                        sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:                     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                             sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                          sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                                   sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                                sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                        sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                                     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                                 sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                                     sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                                  sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                                      sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                                   sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                         sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                                      sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                        sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                              sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                                   sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                                 sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityConsumed:                                                                           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityConsumed,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityRequired:                                                                           sourceContract.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityRequired,
		BodySmokeWeighted:            sourceContract.BodySmokeWeighted,
		NanoDirectRunner:             sourceContract.NanoDirectRunner,
		NanoDirectFinalGate:          sourceContract.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof: sourceContract.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:      sourceContract.BoundaryReportFullChain,
		SourceAuthorityGranted:       sourceContract.SourceAuthorityGranted,
		AuthorityGranted:             false,
		ContractsReady:               false,
		WriteAllowed:                 false,
		AdmissionAllowed:             false,
		LiveAdmissionEnabled:         false,
		MutatesState:                 false,
		BodyTarget:                   "none",
		Passed:                       true,
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger blocked by blocked writer contract; ledger receipt append remains closed",
	}
	inventory.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerCausalID(inventory)
	inventory.AdmissionLedgerHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerHash(inventory)
	inventory.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReadBackHash(inventory)
	inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID(inventory)
	if inventory.CausalID == "" ||
		inventory.AdmissionLedgerHash == "" ||
		inventory.ReadBackHash == "" ||
		inventory.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID == "" ||
		inventory.AdmissionLedgerHash == inventory.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger read-back proof failed")
	}
	raw, err := json.MarshalIndent(inventory, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_report=%s\n", outputPath, writerContractPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger")
	}
	if report.TargetMode != "closed_admission_ledger_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger target_mode mismatch: got %q want %q", report.TargetMode, "closed_admission_ledger_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_writer_contract")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_writer_contract")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.InventoryState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger inventory_state mismatch: got %q want %q", report.InventoryState, "blocked")
	}
	if report.InventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger inventory_action mismatch: got %q want %q", report.InventoryAction, "reject_blocked_writer_preflight")
	}
	if report.ContractState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger contract_state mismatch: got %q want %q", report.ContractState, "blocked")
	}
	if report.ContractAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger contract_action mismatch: got %q want %q", report.ContractAction, "reject_blocked_writer_inventory")
	}
	if report.WriterContract != "none" || report.RollbackContract != "none" || report.AdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger contracts unexpectedly named")
	}
	if report.WriterContractShape != "none" || report.RollbackContractShape != "none" || report.LedgerContractShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger contract shapes unexpectedly named")
	}
	if report.WriteScope != "none" || report.RollbackScope != "none" || report.LedgerMode != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger scopes unexpectedly opened")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger ledger state/action mismatch")
	}
	if report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger append shape unexpectedly opened")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_receipt")
	}
	if report.AdmissionLedgerKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger" ||
		report.AdmissionLedgerMode != "closed_writer_contract_ledger_guard" ||
		report.AdmissionLedgerStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_pre_ledger_append" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReady},
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
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID},
		{"causal_id", report.CausalID},
		{"admission_ledger_hash", report.AdmissionLedgerHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack},
		{"source_writer_contract_reason", report.SourceWriterContractReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack},
		{"source_writer_inventory_reason", report.SourceWriterInventoryReason},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceWriterContractReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract_receipt" ||
		report.SourceWriterContractKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_contract" ||
		report.SourceWriterContractMode != "closed_writer_inventory_contract_guard" ||
		report.SourceWriterContractStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_pre_admission_ledger_writer_contract" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer contract shape mismatch")
	}
	if report.SourceWriterContractContractState != "blocked" ||
		report.SourceWriterContractContractAction != "reject_blocked_writer_inventory" ||
		report.SourceWriterContractWriterAction != "reject_blocked_writer_inventory" ||
		report.SourceWriterContractRollbackAction != "reject_blocked_writer_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer contract state mismatch")
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
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer contract unexpectedly opened")
	}
	if report.SourceWriterContractBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_contract_body_target mismatch: got %q want %q", report.SourceWriterContractBodyTarget, "none")
	}
	if report.SourceWriterContractReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer contract blocked by blocked writer inventory; writer, rollback, and ledger contract shapes remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_contract_reason mismatch: got %q", report.SourceWriterContractReason)
	}
	if report.SourceWriterInventoryReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory_receipt" ||
		report.SourceWriterInventoryKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_inventory" ||
		report.SourceWriterInventoryMode != "closed_writer_preflight_inventory_guard" ||
		report.SourceWriterInventoryStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_pre_writer_contract_inventory" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer inventory shape mismatch")
	}
	if report.SourceWriterInventoryWriterState != "blocked" ||
		report.SourceWriterInventoryWriterAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryRollbackState != "blocked" ||
		report.SourceWriterInventoryRollbackAction != "reject_blocked_writer_preflight" ||
		report.SourceWriterInventoryInventoryState != "blocked" ||
		report.SourceWriterInventoryInventoryAction != "reject_blocked_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer inventory state mismatch")
	}
	if report.SourceWriterInventoryWriterContract != "none" ||
		report.SourceWriterInventoryRollbackContract != "none" ||
		report.SourceWriterInventoryAdmissionLedgerContract != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer inventory contracts unexpectedly named")
	}
	if report.SourceWriterInventoryBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_inventory_body_target mismatch: got %q want %q", report.SourceWriterInventoryBodyTarget, "none")
	}
	if report.SourceWriterInventoryReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer inventory blocked by blocked writer preflight; writer, rollback, and ledger contracts remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_inventory_reason mismatch: got %q", report.SourceWriterInventoryReason)
	}
	if report.SourceWriterPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight_receipt" ||
		report.SourceWriterPreflightKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_writer_preflight" ||
		report.SourceWriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.SourceWriterPreflightStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_pre_writer_inventory_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer preflight shape mismatch")
	}
	if report.SourceWriterPreflightWriterState != "blocked" ||
		report.SourceWriterPreflightWriterAction != "reject_blocked_live_stage" ||
		report.SourceWriterPreflightRollbackState != "blocked" ||
		report.SourceWriterPreflightRollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer preflight state mismatch")
	}
	if report.SourceWriterPreflightBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_preflight_body_target mismatch: got %q want %q", report.SourceWriterPreflightBodyTarget, "none")
	}
	if report.SourceWriterPreflightReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_writer_preflight_reason mismatch: got %q", report.SourceWriterPreflightReason)
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage" ||
		report.SourceLiveStageMode != "closed_switch_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_pre_writer_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source switch state/action not carried")
	}
	if report.SourceSwitchKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_promotion_action mismatch: got %q want empty", report.SourcePromotionAction)
	}
	if report.SourcePromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger causal prefix mismatch")
	}
	if !strings.HasPrefix(report.AdmissionLedgerHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-read-") ||
		report.AdmissionLedgerHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-inventory-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer inventory mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-preflight-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer preflight mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID, "weighted-resonance-graft-admission-final-gate-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger causal_id mismatch")
	}
	if report.AdmissionLedgerHash == "" || report.AdmissionLedgerHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger admission_ledger_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger read_back_hash mismatch")
	}
	if report.AdmissionLedgerHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger id mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-writer-contract-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger source writer contract mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger blocked by blocked writer contract; ledger receipt append remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport) string {
	h := hashJSON(struct {
		SourceWriterContractID   string `json:"source_writer_contract_id"`
		SourceWriterContractRead string `json:"source_writer_contract_read_back_hash"`
		SourceWriterInventoryID  string `json:"source_writer_inventory_id"`
		Target                   string `json:"target"`
		AdmissionLedgerKind      string `json:"admission_ledger_kind"`
		AdmissionLedgerStage     string `json:"admission_ledger_stage"`
	}{
		SourceWriterContractID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID,
		SourceWriterContractRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack,
		SourceWriterInventoryID:  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID,
		Target:                   sw.Target,
		AdmissionLedgerKind:      sw.AdmissionLedgerKind,
		AdmissionLedgerStage:     sw.AdmissionLedgerStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport) string {
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
		SourceWriterContractID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID,
		SourceWriterContractHash:  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash,
		SourceWriterContractRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack,
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID,
		SourceWriterInventoryHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack,
		SourceWriterPreflightID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightID,
		SourceWriterPreflightHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightHash,
		SourceWriterPreflightRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterPreflightReadBack,
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
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport) string {
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
		SourceWriterContractID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID,
		SourceWriterContractRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack,
		AdmissionLedgerKind:      sw.AdmissionLedgerKind,
		AdmissionLedgerReady:     sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady,
		WriterContractConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractConsumed,
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
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport) string {
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
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_live_stage_ledger"`
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
		SourceWriterContractID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractID,
		SourceWriterContractHash:  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractHash,
		SourceWriterContractRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterContractReadBack,
		SourceWriterInventoryID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryID,
		SourceWriterInventoryHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryHash,
		SourceWriterInventoryRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageWriterInventoryReadBack,
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
		AdmissionLedgerHash:       sw.AdmissionLedgerHash,
		ReadBackHash:              sw.ReadBackHash,
		Ready:                     sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReady,
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
		NextStepBlockedWithout:    sw.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedger,
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
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-live-stage-ledger-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateLiveStageLedgerReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate live stage ledger decode failed: %w", err)
	}
	return report, root, nil
}
