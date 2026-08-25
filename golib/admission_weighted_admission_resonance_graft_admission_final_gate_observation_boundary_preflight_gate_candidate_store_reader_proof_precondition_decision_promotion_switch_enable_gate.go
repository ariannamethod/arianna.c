package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport struct {
	Schema                                                                                                                                                       string `json:"schema"`
	Status                                                                                                                                                       string `json:"status"`
	Target                                                                                                                                                       string `json:"target"`
	TargetKind                                                                                                                                                   string `json:"target_kind"`
	TargetMode                                                                                                                                                   string `json:"target_mode"`
	Action                                                                                                                                                       string `json:"action"`
	EnableState                                                                                                                                                  string `json:"enable_state"`
	EnableAction                                                                                                                                                 string `json:"enable_action"`
	SwitchState                                                                                                                                                  string `json:"switch_state"`
	SwitchAction                                                                                                                                                 string `json:"switch_action"`
	Promotion                                                                                                                                                    string `json:"promotion"`
	LedgerState                                                                                                                                                  string `json:"ledger_state"`
	LedgerAction                                                                                                                                                 string `json:"ledger_action"`
	LedgerContract                                                                                                                                               string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                                             string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                                           string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                                             string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                                                  bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                                          bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchConsumed        bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchRequired        bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id"`
	ReceiptShape                                                                                                                                                 string `json:"receipt_shape"`
	EnableGateKind                                                                                                                                               string `json:"enable_gate_kind"`
	EnableGateMode                                                                                                                                               string `json:"enable_gate_mode"`
	EnableGateStage                                                                                                                                              string `json:"enable_gate_stage"`
	CausalID                                                                                                                                                     string `json:"causal_id"`
	EnableGateHash                                                                                                                                               string `json:"enable_gate_hash"`
	ReadBackHash                                                                                                                                                 string `json:"read_back_hash"`
	SwitchVerified                                                                                                                                               bool   `json:"switch_verified"`
	SwitchHashVerified                                                                                                                                           bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                                                                                                                       bool   `json:"switch_read_back_verified"`
	PromotionVerified                                                                                                                                            bool   `json:"promotion_verified"`
	PromotionHashVerified                                                                                                                                        bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                                                                                                                    bool   `json:"promotion_read_back_verified"`
	DecisionVerified                                                                                                                                             bool   `json:"decision_verified"`
	DecisionHashVerified                                                                                                                                         bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                                                                                                                     bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                                                                                                                    bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                                                     bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                                                 bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                                                bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                                            bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                                                        bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                                          bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                                                bool   `json:"store_verified"`
	CandidateVerified                                                                                                                                            bool   `json:"candidate_verified"`
	GateVerified                                                                                                                                                 bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                                            bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                                             bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                                          bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                                             bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                                               bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                                            bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                                                 bool   `json:"seal_verified"`
	PermitVerified                                                                                                                                               bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                                            bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                                           bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                                         bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                                                       bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                                            bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                                                        bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                                            bool   `json:"admission_required"`
	ShadowOnly                                                                                                                                                   bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                                                 bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                                                   bool   `json:"dry_run_only"`
	LiveReady                                                                                                                                                    bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                                          bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                                         bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                                                        bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                                          bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                                          bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                                          bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                                          bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                                             bool   `json:"rollback_required"`
	ReadOnly                                                                                                                                                     bool   `json:"read_only"`
	ReplayOnly                                                                                                                                                   bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                                             bool   `json:"authority_granted"`
	ContractsReady                                                                                                                                               bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                                                 bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                                             bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                                         bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                                                 bool   `json:"mutates_state"`
	BodyTarget                                                                                                                                                   string `json:"body_target"`
	Passed                                                                                                                                                       bool   `json:"passed"`
	Reason                                                                                                                                                       string `json:"reason"`

	SourceSchema                                                                                                                                                string `json:"source_schema"`
	SourceStatus                                                                                                                                                string `json:"source_status"`
	SourceTarget                                                                                                                                                string `json:"source_target"`
	SourceReport                                                                                                                                                string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_read_back_hash"`
	SourceSwitchState                                                                                                                                           string `json:"source_switch_state"`
	SourceSwitchAction                                                                                                                                          string `json:"source_switch_action"`
	SourceSwitchReceiptShape                                                                                                                                    string `json:"source_switch_receipt_shape"`
	SourceSwitchKind                                                                                                                                            string `json:"source_switch_kind"`
	SourceSwitchMode                                                                                                                                            string `json:"source_switch_mode"`
	SourceSwitchStage                                                                                                                                           string `json:"source_switch_stage"`
	SourceSwitchLedgerReady                                                                                                                                     bool   `json:"source_switch_ledger_ready"`
	SourceSwitchLedgerAppendAllowed                                                                                                                             bool   `json:"source_switch_ledger_append_allowed"`
	SourceSwitchAdmissionRequired                                                                                                                               bool   `json:"source_switch_admission_required"`
	SourceSwitchShadowOnly                                                                                                                                      bool   `json:"source_switch_shadow_only"`
	SourceSwitchGraftAllowed                                                                                                                                    bool   `json:"source_switch_graft_allowed"`
	SourceSwitchDryRunOnly                                                                                                                                      bool   `json:"source_switch_dry_run_only"`
	SourceSwitchLiveReady                                                                                                                                       bool   `json:"source_switch_live_ready"`
	SourceSwitchRawDreamTextAllowed                                                                                                                             bool   `json:"source_switch_raw_dream_text_allowed"`
	SourceSwitchRawDreamTextObserved                                                                                                                            bool   `json:"source_switch_raw_dream_text_observed"`
	SourceSwitchRawDreamTextForwarded                                                                                                                           bool   `json:"source_switch_raw_dream_text_forwarded"`
	SourceSwitchJanusSurfaceAllowed                                                                                                                             bool   `json:"source_switch_janus_surface_allowed"`
	SourceSwitchCoocLearningAllowed                                                                                                                             bool   `json:"source_switch_cooc_learning_allowed"`
	SourceSwitchDeltaHarvestAllowed                                                                                                                             bool   `json:"source_switch_delta_harvest_allowed"`
	SourceSwitchBodyMutationAllowed                                                                                                                             bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchRollbackRequired                                                                                                                                bool   `json:"source_switch_rollback_required"`
	SourceSwitchReadOnly                                                                                                                                        bool   `json:"source_switch_read_only"`
	SourceSwitchReplayOnly                                                                                                                                      bool   `json:"source_switch_replay_only"`
	SourceSwitchAuthorityGranted                                                                                                                                bool   `json:"source_switch_authority_granted"`
	SourceSwitchContractsReady                                                                                                                                  bool   `json:"source_switch_contracts_ready"`
	SourceSwitchWriteAllowed                                                                                                                                    bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                                                                                                                bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                                                                                                            bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchMutatesState                                                                                                                                    bool   `json:"source_switch_mutates_state"`
	SourceSwitchBodyTarget                                                                                                                                      string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                                                                                                          bool   `json:"source_switch_passed"`
	SourceSwitchReason                                                                                                                                          string `json:"source_switch_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_read_back_hash"`
	SourcePromotion                                                                                                                                             string `json:"source_promotion"`
	SourcePromotionAction                                                                                                                                       string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                                                                                                                 string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                                                                                                         string `json:"source_promotion_kind"`
	SourcePromotionMode                                                                                                                                         string `json:"source_promotion_mode"`
	SourcePromotionStage                                                                                                                                        string `json:"source_promotion_stage"`
	SourcePromotionLedgerReady                                                                                                                                  bool   `json:"source_promotion_ledger_ready"`
	SourcePromotionLedgerAppendAllowed                                                                                                                          bool   `json:"source_promotion_ledger_append_allowed"`
	SourcePromotionGraftAllowed                                                                                                                                 bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionWriteAllowed                                                                                                                                 bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                                                                                                             bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                                                                                                         bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionBodyMutationAllowed                                                                                                                          bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionBodyTarget                                                                                                                                   string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                                                                                                       bool   `json:"source_promotion_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID                      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash                    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_read_back_hash"`
	SourceDecision                                                                                                                                              string `json:"source_decision"`
	SourceDecisionKind                                                                                                                                          string `json:"source_decision_kind"`
	SourceDecisionMode                                                                                                                                          string `json:"source_decision_mode"`
	SourceDecisionLedgerAppendAllowed                                                                                                                           bool   `json:"source_decision_ledger_append_allowed"`
	SourceDecisionGraftAllowed                                                                                                                                  bool   `json:"source_decision_graft_allowed"`
	SourceDecisionWriteAllowed                                                                                                                                  bool   `json:"source_decision_write_allowed"`
	SourceDecisionLiveAdmissionEnabled                                                                                                                          bool   `json:"source_decision_live_admission_enabled"`
	SourceDecisionBodyMutationAllowed                                                                                                                           bool   `json:"source_decision_body_mutation_allowed"`
	SourceDecisionBodyTarget                                                                                                                                    string `json:"source_decision_body_target"`
	SourceDecisionPassed                                                                                                                                        bool   `json:"source_decision_passed"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID                              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady                           bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady                                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                                               string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                                            bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                                                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                                                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                                          string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                                                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                                               bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                                                       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                                                    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                                                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                                           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                                                        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                                          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                                                     bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                                                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                                                                                                bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                                               bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                                               bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                                                     bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                                            bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                                            bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                                        bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                                                      bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                                                      bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                                              bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                                            bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_PROMOTION_SWITCH_ENABLE_GATE_REPORT")
	}
	switchPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate output path missing")
	}
	sw, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportForAssert(switchPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReportError(sw, root); err != nil {
		return err
	}
	gate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport{
		Schema:         admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateSchema,
		Status:         "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_disabled_dry_run",
		Target:         "live_route_admission_next_step",
		TargetKind:     "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate",
		TargetMode:     "closed_enable_gate_dry_run",
		Action:         "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run",
		EnableState:    "disabled",
		EnableAction:   "require_operator_key",
		SwitchState:    sw.SwitchState,
		SwitchAction:   sw.SwitchAction,
		Promotion:      sw.Promotion,
		LedgerState:    "blocked",
		LedgerAction:   "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ledger_append",
		LedgerContract: "none", LedgerEntrypoint: "none", LedgerReceiptShape: "none", LedgerWriteScope: "none",
		LedgerReady: false, LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchConsumed:        true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchRequired:        true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate: true,
		ReceiptShape:                 "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_receipt",
		EnableGateKind:               "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate",
		EnableGateMode:               "closed_switch_enable_guard",
		EnableGateStage:              "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_pre_live_admission_enable_gate",
		SwitchVerified:               true,
		SwitchHashVerified:           true,
		SwitchReadBackVerified:       true,
		PromotionVerified:            sw.PromotionVerified,
		PromotionHashVerified:        sw.PromotionHashVerified,
		PromotionReadBackVerified:    sw.PromotionReadBackVerified,
		DecisionVerified:             sw.DecisionVerified,
		DecisionHashVerified:         sw.DecisionHashVerified,
		DecisionReadBackVerified:     sw.DecisionReadBackVerified,
		ProofPreconditionVerified:    sw.ProofPreconditionVerified,
		PreconditionHashVerified:     sw.PreconditionHashVerified,
		PreconditionReadBackVerified: sw.PreconditionReadBackVerified,
		ProofVerified:                sw.ProofVerified,
		ProofHashVerified:            sw.ProofHashVerified,
		ProofReadBackVerified:        sw.ProofReadBackVerified,
		StoreReaderVerified:          sw.StoreReaderVerified,
		StoreVerified:                sw.StoreVerified,
		CandidateVerified:            sw.CandidateVerified,
		GateVerified:                 sw.GateVerified,
		PreflightVerified:            sw.PreflightVerified,
		BoundaryVerified:             sw.BoundaryVerified,
		ObservationVerified:          sw.ObservationVerified,
		ReceiverVerified:             sw.ReceiverVerified,
		IntentVerified:               sw.IntentVerified,
		FinalGateVerified:            sw.FinalGateVerified,
		SealVerified:                 sw.SealVerified,
		PermitVerified:               sw.PermitVerified,
		AuthorityVerified:            sw.AuthorityVerified,
		ReaderHashVerified:           sw.ReaderHashVerified,
		ReaderReplayVerified:         sw.ReaderReplayVerified,
		ReaderReadBackVerified:       sw.ReaderReadBackVerified,
		StoreHashVerified:            sw.StoreHashVerified,
		StoreReadBackVerified:        sw.StoreReadBackVerified,
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
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch held behind disabled enable gate; operator key absent and mutation refused",

		SourceSchema: sw.Schema, SourceStatus: sw.Status, SourceTarget: sw.Target, SourceReport: switchPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID:       sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady:    sw.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchCausalID: sw.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash:     sw.SwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack: sw.ReadBackHash,
		SourceSwitchState: sw.SwitchState, SourceSwitchAction: sw.SwitchAction, SourceSwitchReceiptShape: sw.ReceiptShape, SourceSwitchKind: sw.SwitchKind, SourceSwitchMode: sw.SwitchMode, SourceSwitchStage: sw.SwitchStage,
		SourceSwitchLedgerReady: sw.LedgerReady, SourceSwitchLedgerAppendAllowed: sw.LedgerAppendAllowed,
		SourceSwitchAdmissionRequired: sw.AdmissionRequired, SourceSwitchShadowOnly: sw.ShadowOnly, SourceSwitchGraftAllowed: sw.GraftAllowed, SourceSwitchDryRunOnly: sw.DryRunOnly, SourceSwitchLiveReady: sw.LiveReady,
		SourceSwitchRawDreamTextAllowed: sw.RawDreamTextAllowed, SourceSwitchRawDreamTextObserved: sw.RawDreamTextObserved, SourceSwitchRawDreamTextForwarded: sw.RawDreamTextForwarded, SourceSwitchJanusSurfaceAllowed: sw.JanusSurfaceAllowed, SourceSwitchCoocLearningAllowed: sw.CoocLearningAllowed, SourceSwitchDeltaHarvestAllowed: sw.DeltaHarvestAllowed, SourceSwitchBodyMutationAllowed: sw.BodyMutationAllowed,
		SourceSwitchRollbackRequired: sw.RollbackRequired, SourceSwitchReadOnly: sw.ReadOnly, SourceSwitchReplayOnly: sw.ReplayOnly, SourceSwitchAuthorityGranted: sw.AuthorityGranted, SourceSwitchContractsReady: sw.ContractsReady, SourceSwitchWriteAllowed: sw.WriteAllowed, SourceSwitchAdmissionAllowed: sw.AdmissionAllowed, SourceSwitchLiveAdmissionEnabled: sw.LiveAdmissionEnabled, SourceSwitchMutatesState: sw.MutatesState, SourceSwitchBodyTarget: sw.BodyTarget, SourceSwitchPassed: sw.Passed, SourceSwitchReason: sw.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash:     sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReadBack,
		SourcePromotion: sw.SourcePromotion, SourcePromotionAction: sw.SourcePromotionAction, SourcePromotionReceiptShape: sw.SourcePromotionReceiptShape, SourcePromotionKind: sw.SourcePromotionKind, SourcePromotionMode: sw.SourcePromotionMode, SourcePromotionStage: sw.SourcePromotionStage,
		SourcePromotionLedgerReady: sw.SourcePromotionLedgerReady, SourcePromotionLedgerAppendAllowed: sw.SourcePromotionLedgerAppendAllowed, SourcePromotionGraftAllowed: sw.SourcePromotionGraftAllowed, SourcePromotionWriteAllowed: sw.SourcePromotionWriteAllowed, SourcePromotionAdmissionAllowed: sw.SourcePromotionAdmissionAllowed, SourcePromotionLiveAdmissionEnabled: sw.SourcePromotionLiveAdmissionEnabled, SourcePromotionBodyMutationAllowed: sw.SourcePromotionBodyMutationAllowed, SourcePromotionBodyTarget: sw.SourcePromotionBodyTarget, SourcePromotionPassed: sw.SourcePromotionPassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash:     sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBack,
		SourceDecision: sw.SourceDecision, SourceDecisionKind: sw.SourceDecisionKind, SourceDecisionMode: sw.SourceDecisionMode, SourceDecisionLedgerAppendAllowed: sw.SourceDecisionLedgerAppendAllowed, SourceDecisionGraftAllowed: sw.SourceDecisionGraftAllowed, SourceDecisionWriteAllowed: sw.SourceDecisionWriteAllowed, SourceDecisionLiveAdmissionEnabled: sw.SourceDecisionLiveAdmissionEnabled, SourceDecisionBodyMutationAllowed: sw.SourceDecisionBodyMutationAllowed, SourceDecisionBodyTarget: sw.SourceDecisionBodyTarget, SourceDecisionPassed: sw.SourceDecisionPassed,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady: sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:                sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:             sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:                     sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:                  sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:                           sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:                        sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:                                sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:                             sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                                         sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                                      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                                                     sw.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                                             sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                                          sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                                      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                                   sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                                              sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                                           sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                                 sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                                              sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                                                sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                                      sw.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                                           sw.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                                                                      sw.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                                         sw.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                                                                      sw.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified: sw.SourceWriterInventoryVerified, SourceWriterPreflightVerified: sw.SourceWriterPreflightVerified, SourceAdmissionRequired: sw.SourceAdmissionRequired, SourceShadowOnly: sw.SourceShadowOnly, SourceDryRunOnly: sw.SourceDryRunOnly, SourceRequiresWriter: sw.SourceRequiresWriter, SourceRollbackRequired: sw.SourceRollbackRequired, SourceRequiresRollback: sw.SourceRequiresRollback, SourceReadOnly: sw.SourceReadOnly, SourceReplayOnly: sw.SourceReplayOnly,
	}
	gate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID(gate)
	gate.EnableGateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash(gate)
	gate.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBackHash(gate)
	gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID(gate)
	if gate.CausalID == "" || gate.EnableGateHash == "" || gate.ReadBackHash == "" || gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID == "" || gate.EnableGateHash == gate.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(gate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_report=%s\n", outputPath, switchPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_disabled_dry_run")
	}
	if report.Target != "live_route_admission_next_step" || report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate" || report.TargetMode != "closed_enable_gate_dry_run" || report.Action != "hold_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate route shape mismatch")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" || report.SwitchAction != "hold_pending_live_admission" || report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate switch state mismatch")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ledger_append" || report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_receipt" || report.EnableGateKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate" || report.EnableGateMode != "closed_switch_enable_guard" || report.EnableGateStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_pre_live_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchConsumed}, {"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchRequired}, {"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate},
		{"switch_verified", report.SwitchVerified}, {"switch_hash_verified", report.SwitchHashVerified}, {"switch_read_back_verified", report.SwitchReadBackVerified}, {"promotion_verified", report.PromotionVerified}, {"promotion_hash_verified", report.PromotionHashVerified}, {"promotion_read_back_verified", report.PromotionReadBackVerified}, {"decision_verified", report.DecisionVerified}, {"decision_hash_verified", report.DecisionHashVerified}, {"decision_read_back_verified", report.DecisionReadBackVerified}, {"proof_precondition_verified", report.ProofPreconditionVerified}, {"precondition_hash_verified", report.PreconditionHashVerified}, {"precondition_read_back_verified", report.PreconditionReadBackVerified}, {"proof_verified", report.ProofVerified}, {"proof_hash_verified", report.ProofHashVerified}, {"proof_read_back_verified", report.ProofReadBackVerified}, {"store_reader_verified", report.StoreReaderVerified}, {"store_verified", report.StoreVerified}, {"candidate_verified", report.CandidateVerified}, {"gate_verified", report.GateVerified}, {"preflight_verified", report.PreflightVerified}, {"boundary_verified", report.BoundaryVerified}, {"observation_verified", report.ObservationVerified}, {"receiver_verified", report.ReceiverVerified}, {"intent_verified", report.IntentVerified}, {"final_gate_verified", report.FinalGateVerified}, {"seal_verified", report.SealVerified}, {"permit_verified", report.PermitVerified}, {"authority_verified", report.AuthorityVerified}, {"reader_hash_verified", report.ReaderHashVerified}, {"reader_replay_verified", report.ReaderReplayVerified}, {"reader_read_back_verified", report.ReaderReadBackVerified}, {"store_hash_verified", report.StoreHashVerified}, {"store_read_back_verified", report.StoreReadBackVerified}, {"admission_required", report.AdmissionRequired}, {"shadow_only", report.ShadowOnly}, {"dry_run_only", report.DryRunOnly}, {"live_ready", report.LiveReady}, {"rollback_required", report.RollbackRequired}, {"read_only", report.ReadOnly}, {"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReady}, {"source_switch_admission_required", report.SourceSwitchAdmissionRequired}, {"source_switch_shadow_only", report.SourceSwitchShadowOnly}, {"source_switch_dry_run_only", report.SourceSwitchDryRunOnly}, {"source_switch_live_ready", report.SourceSwitchLiveReady}, {"source_switch_rollback_required", report.SourceSwitchRollbackRequired}, {"source_switch_read_only", report.SourceSwitchReadOnly}, {"source_switch_replay_only", report.SourceSwitchReplayOnly}, {"source_switch_passed", report.SourceSwitchPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionReady}, {"source_promotion_passed", report.SourcePromotionPassed}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady}, {"source_decision_passed", report.SourceDecisionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady}, {"source_writer_inventory_verified", report.SourceWriterInventoryVerified}, {"source_writer_preflight_verified", report.SourceWriterPreflightVerified}, {"source_admission_required", report.SourceAdmissionRequired}, {"source_shadow_only", report.SourceShadowOnly}, {"source_dry_run_only", report.SourceDryRunOnly}, {"source_requires_writer", report.SourceRequiresWriter}, {"source_rollback_required", report.SourceRollbackRequired}, {"source_requires_rollback", report.SourceRequiresRollback}, {"source_read_only", report.SourceReadOnly}, {"source_replay_only", report.SourceReplayOnly}, {"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady}, {"ledger_append_allowed", report.LedgerAppendAllowed}, {"graft_allowed", report.GraftAllowed}, {"raw_dream_text_allowed", report.RawDreamTextAllowed}, {"raw_dream_text_observed", report.RawDreamTextObserved}, {"raw_dream_text_forwarded", report.RawDreamTextForwarded}, {"janus_surface_allowed", report.JanusSurfaceAllowed}, {"cooc_learning_allowed", report.CoocLearningAllowed}, {"delta_harvest_allowed", report.DeltaHarvestAllowed}, {"body_mutation_allowed", report.BodyMutationAllowed}, {"authority_granted", report.AuthorityGranted}, {"contracts_ready", report.ContractsReady}, {"write_allowed", report.WriteAllowed}, {"admission_allowed", report.AdmissionAllowed}, {"live_admission_enabled", report.LiveAdmissionEnabled}, {"mutates_state", report.MutatesState},
		{"source_switch_ledger_ready", report.SourceSwitchLedgerReady}, {"source_switch_ledger_append_allowed", report.SourceSwitchLedgerAppendAllowed}, {"source_switch_graft_allowed", report.SourceSwitchGraftAllowed}, {"source_switch_raw_dream_text_allowed", report.SourceSwitchRawDreamTextAllowed}, {"source_switch_body_mutation_allowed", report.SourceSwitchBodyMutationAllowed}, {"source_switch_authority_granted", report.SourceSwitchAuthorityGranted}, {"source_switch_contracts_ready", report.SourceSwitchContractsReady}, {"source_switch_write_allowed", report.SourceSwitchWriteAllowed}, {"source_switch_admission_allowed", report.SourceSwitchAdmissionAllowed}, {"source_switch_live_admission_enabled", report.SourceSwitchLiveAdmissionEnabled}, {"source_switch_mutates_state", report.SourceSwitchMutatesState},
		{"source_promotion_ledger_ready", report.SourcePromotionLedgerReady}, {"source_promotion_ledger_append_allowed", report.SourcePromotionLedgerAppendAllowed}, {"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed}, {"source_promotion_write_allowed", report.SourcePromotionWriteAllowed}, {"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed}, {"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled}, {"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed},
		{"source_decision_ledger_append_allowed", report.SourceDecisionLedgerAppendAllowed}, {"source_decision_graft_allowed", report.SourceDecisionGraftAllowed}, {"source_decision_write_allowed", report.SourceDecisionWriteAllowed}, {"source_decision_live_admission_enabled", report.SourceDecisionLiveAdmissionEnabled}, {"source_decision_body_mutation_allowed", report.SourceDecisionBodyMutationAllowed}, {"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct{ name, value string }{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID}, {"causal_id", report.CausalID}, {"enable_gate_hash", report.EnableGateHash}, {"read_back_hash", report.ReadBackHash}, {"source_report", report.SourceReport}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_disabled_dry_run" || report.SourceTarget != "live_route_admission_next_step" || report.SourceSwitchState != "disabled" || report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate source switch route mismatch")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_receipt" || report.SourceSwitchKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch" || report.SourceSwitchMode != "closed_promotion_switch_guard" || report.SourceSwitchStage != "post_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_pre_live_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate source switch shape mismatch")
	}
	if report.SourcePromotion != "pending_live_admission" || report.SourcePromotionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion" || report.SourceDecision != "shadow_ready" || report.SourceDecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate source chain shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceSwitchBodyTarget != "none" || report.SourcePromotionBodyTarget != "none" || report.SourceDecisionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-") || !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-causal-") || !strings.HasPrefix(report.EnableGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-") || !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-read-") || report.EnableGateHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate causal_id mismatch")
	}
	if report.EnableGateHash == "" || report.EnableGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate enable_gate_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate read_back_hash mismatch")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch held behind disabled enable gate; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateCausalID(gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport) string {
	h := hashJSON(map[string]interface{}{"source_switch_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "source_switch_read_back_hash": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack, "source_promotion_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_decision_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "switch_state": gate.SwitchState, "switch_action": gate.SwitchAction, "enable_gate_kind": gate.EnableGateKind, "enable_gate_stage": gate.EnableGateStage})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateHash(gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport) string {
	h := hashJSON(map[string]interface{}{"causal_id": gate.CausalID, "source_switch_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "source_switch_hash": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchHash, "source_switch_read_back_hash": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchReadBack, "enable_state": gate.EnableState, "enable_action": gate.EnableAction, "switch_state": gate.SwitchState, "switch_action": gate.SwitchAction, "promotion": gate.Promotion, "enable_gate_mode": gate.EnableGateMode, "receipt_shape": gate.ReceiptShape, "switch_verified": gate.SwitchVerified, "switch_hash_verified": gate.SwitchHashVerified, "switch_read_back_verified": gate.SwitchReadBackVerified, "read_only": gate.ReadOnly, "replay_only": gate.ReplayOnly, "admission_required": gate.AdmissionRequired, "shadow_only": gate.ShadowOnly, "dry_run_only": gate.DryRunOnly, "graft_allowed": gate.GraftAllowed, "ledger_append_allowed": gate.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReadBackHash(gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport) string {
	h := hashJSON(map[string]interface{}{"enable_gate_hash": gate.EnableGateHash, "source_switch_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "enable_state": gate.EnableState, "enable_action": gate.EnableAction, "switch_state": gate.SwitchState, "enable_gate_ready": gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady, "body_mutation": gate.BodyMutationAllowed, "live_admission": gate.LiveAdmissionEnabled, "write_allowed": gate.WriteAllowed, "admission_allowed": gate.AdmissionAllowed, "ledger_append_allowed": gate.LedgerAppendAllowed})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateID(gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport) string {
	h := hashJSON(map[string]interface{}{"schema": gate.Schema, "status": gate.Status, "action": gate.Action, "enable_state": gate.EnableState, "enable_action": gate.EnableAction, "switch_state": gate.SwitchState, "switch_action": gate.SwitchAction, "promotion": gate.Promotion, "source_report": gate.SourceReport, "source_switch_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchID, "source_promotion_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionID, "source_decision_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "source_precondition_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "source_proof_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "source_reader_id": gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "causal_id": gate.CausalID, "enable_gate_hash": gate.EnableGateHash, "read_back_hash": gate.ReadBackHash, "receipt_shape": gate.ReceiptShape, "enable_gate_kind": gate.EnableGateKind, "enable_gate_mode": gate.EnableGateMode, "enable_gate_stage": gate.EnableGateStage, "body_target": gate.BodyTarget, "ready": gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReady, "switch_verified": gate.SwitchVerified, "switch_hash_verified": gate.SwitchHashVerified, "switch_read_back_verified": gate.SwitchReadBackVerified, "admission_required": gate.AdmissionRequired, "shadow_only": gate.ShadowOnly, "graft_allowed": gate.GraftAllowed, "dry_run_only": gate.DryRunOnly, "read_only": gate.ReadOnly, "replay_only": gate.ReplayOnly, "live_ready": gate.LiveReady, "contracts_ready": gate.ContractsReady, "write_allowed": gate.WriteAllowed, "admission_allowed": gate.AdmissionAllowed, "live_admission_enabled": gate.LiveAdmissionEnabled, "mutates_state": gate.MutatesState, "ledger_append_allowed": gate.LedgerAppendAllowed, "next_step_blocked_without": gate.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGate})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-promotion-switch-enable-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionPromotionSwitchEnableGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision promotion switch enable gate decode failed: %w", err)
	}
	return report, root, nil
}
