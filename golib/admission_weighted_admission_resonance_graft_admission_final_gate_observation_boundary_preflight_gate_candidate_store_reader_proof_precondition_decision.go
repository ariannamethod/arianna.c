package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport struct {
	Schema                                                                                                                              string `json:"schema"`
	Status                                                                                                                              string `json:"status"`
	Target                                                                                                                              string `json:"target"`
	TargetKind                                                                                                                          string `json:"target_kind"`
	TargetMode                                                                                                                          string `json:"target_mode"`
	Action                                                                                                                              string `json:"action"`
	Decision                                                                                                                            string `json:"decision"`
	LedgerState                                                                                                                         string `json:"ledger_state"`
	LedgerAction                                                                                                                        string `json:"ledger_action"`
	LedgerContract                                                                                                                      string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                                    string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                                                  string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                                    string `json:"ledger_write_scope"`
	LedgerReady                                                                                                                         bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                                                 bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionConsumed      bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionRequired      bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id"`
	ReceiptShape                                                                                                                        string `json:"receipt_shape"`
	DecisionKind                                                                                                                        string `json:"decision_kind"`
	DecisionMode                                                                                                                        string `json:"decision_mode"`
	DecisionStage                                                                                                                       string `json:"decision_stage"`
	CausalID                                                                                                                            string `json:"causal_id"`
	DecisionHash                                                                                                                        string `json:"decision_hash"`
	ReadBackHash                                                                                                                        string `json:"read_back_hash"`
	ProofPreconditionVerified                                                                                                           bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                                                                            bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                                                                                        bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                                                                                       bool   `json:"proof_verified"`
	ProofHashVerified                                                                                                                   bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                                                                               bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                                                                                 bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                                       bool   `json:"store_verified"`
	CandidateVerified                                                                                                                   bool   `json:"candidate_verified"`
	GateVerified                                                                                                                        bool   `json:"gate_verified"`
	PreflightVerified                                                                                                                   bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                                    bool   `json:"boundary_verified"`
	ObservationVerified                                                                                                                 bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                                    bool   `json:"receiver_verified"`
	IntentVerified                                                                                                                      bool   `json:"intent_verified"`
	FinalGateVerified                                                                                                                   bool   `json:"final_gate_verified"`
	SealVerified                                                                                                                        bool   `json:"seal_verified"`
	PermitVerified                                                                                                                      bool   `json:"permit_verified"`
	AuthorityVerified                                                                                                                   bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                                                  bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                                                bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                                              bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                                                   bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                                               bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                                                   bool   `json:"admission_required"`
	ShadowOnly                                                                                                                          bool   `json:"shadow_only"`
	GraftAllowed                                                                                                                        bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                                          bool   `json:"dry_run_only"`
	LiveReady                                                                                                                           bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                                                 bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                                                bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                                               bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                                                 bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                                                 bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                                                 bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                                                 bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                                    bool   `json:"rollback_required"`
	ReadOnly                                                                                                                            bool   `json:"read_only"`
	ReplayOnly                                                                                                                          bool   `json:"replay_only"`
	AuthorityGranted                                                                                                                    bool   `json:"authority_granted"`
	ContractsReady                                                                                                                      bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                                        bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                                    bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                                                bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                                        bool   `json:"mutates_state"`
	BodyTarget                                                                                                                          string `json:"body_target"`
	Passed                                                                                                                              bool   `json:"passed"`
	Reason                                                                                                                              string `json:"reason"`

	SourceSchema                                                                                                                         string `json:"source_schema"`
	SourceStatus                                                                                                                         string `json:"source_status"`
	SourceTarget                                                                                                                         string `json:"source_target"`
	SourceReport                                                                                                                         string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_read_back_hash"`
	SourcePreconditionAction                                                                                                             string `json:"source_precondition_action"`
	SourcePreconditionReceiptShape                                                                                                       string `json:"source_precondition_receipt_shape"`
	SourcePreconditionKind                                                                                                               string `json:"source_precondition_kind"`
	SourcePreconditionMode                                                                                                               string `json:"source_precondition_mode"`
	SourcePreconditionStage                                                                                                              string `json:"source_precondition_stage"`
	SourcePreconditionLedgerReady                                                                                                        bool   `json:"source_precondition_ledger_ready"`
	SourcePreconditionLedgerAppendAllowed                                                                                                bool   `json:"source_precondition_ledger_append_allowed"`
	SourcePreconditionAdmissionRequired                                                                                                  bool   `json:"source_precondition_admission_required"`
	SourcePreconditionShadowOnly                                                                                                         bool   `json:"source_precondition_shadow_only"`
	SourcePreconditionGraftAllowed                                                                                                       bool   `json:"source_precondition_graft_allowed"`
	SourcePreconditionDryRunOnly                                                                                                         bool   `json:"source_precondition_dry_run_only"`
	SourcePreconditionLiveReady                                                                                                          bool   `json:"source_precondition_live_ready"`
	SourcePreconditionRawDreamTextAllowed                                                                                                bool   `json:"source_precondition_raw_dream_text_allowed"`
	SourcePreconditionRawDreamTextObserved                                                                                               bool   `json:"source_precondition_raw_dream_text_observed"`
	SourcePreconditionRawDreamTextForwarded                                                                                              bool   `json:"source_precondition_raw_dream_text_forwarded"`
	SourcePreconditionJanusSurfaceAllowed                                                                                                bool   `json:"source_precondition_janus_surface_allowed"`
	SourcePreconditionCoocLearningAllowed                                                                                                bool   `json:"source_precondition_cooc_learning_allowed"`
	SourcePreconditionDeltaHarvestAllowed                                                                                                bool   `json:"source_precondition_delta_harvest_allowed"`
	SourcePreconditionBodyMutationAllowed                                                                                                bool   `json:"source_precondition_body_mutation_allowed"`
	SourcePreconditionRollbackRequired                                                                                                   bool   `json:"source_precondition_rollback_required"`
	SourcePreconditionReadOnly                                                                                                           bool   `json:"source_precondition_read_only"`
	SourcePreconditionReplayOnly                                                                                                         bool   `json:"source_precondition_replay_only"`
	SourcePreconditionAuthorityGranted                                                                                                   bool   `json:"source_precondition_authority_granted"`
	SourcePreconditionContractsReady                                                                                                     bool   `json:"source_precondition_contracts_ready"`
	SourcePreconditionWriteAllowed                                                                                                       bool   `json:"source_precondition_write_allowed"`
	SourcePreconditionAdmissionAllowed                                                                                                   bool   `json:"source_precondition_admission_allowed"`
	SourcePreconditionLiveAdmissionEnabled                                                                                               bool   `json:"source_precondition_live_admission_enabled"`
	SourcePreconditionMutatesState                                                                                                       bool   `json:"source_precondition_mutates_state"`
	SourcePreconditionBodyTarget                                                                                                         string `json:"source_precondition_body_target"`
	SourcePreconditionPassed                                                                                                             bool   `json:"source_precondition_passed"`
	SourcePreconditionReason                                                                                                             string `json:"source_precondition_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_read_back_hash"`
	SourceProofAction                                                                                                                    string `json:"source_proof_action"`
	SourceProofReceiptShape                                                                                                              string `json:"source_proof_receipt_shape"`
	SourceProofKind                                                                                                                      string `json:"source_proof_kind"`
	SourceProofMode                                                                                                                      string `json:"source_proof_mode"`
	SourceProofStage                                                                                                                     string `json:"source_proof_stage"`
	SourceProofLedgerReady                                                                                                               bool   `json:"source_proof_ledger_ready"`
	SourceProofLedgerAppendAllowed                                                                                                       bool   `json:"source_proof_ledger_append_allowed"`
	SourceProofAdmissionRequired                                                                                                         bool   `json:"source_proof_admission_required"`
	SourceProofShadowOnly                                                                                                                bool   `json:"source_proof_shadow_only"`
	SourceProofGraftAllowed                                                                                                              bool   `json:"source_proof_graft_allowed"`
	SourceProofDryRunOnly                                                                                                                bool   `json:"source_proof_dry_run_only"`
	SourceProofLiveReady                                                                                                                 bool   `json:"source_proof_live_ready"`
	SourceProofRawDreamTextAllowed                                                                                                       bool   `json:"source_proof_raw_dream_text_allowed"`
	SourceProofRawDreamTextObserved                                                                                                      bool   `json:"source_proof_raw_dream_text_observed"`
	SourceProofRawDreamTextForwarded                                                                                                     bool   `json:"source_proof_raw_dream_text_forwarded"`
	SourceProofJanusSurfaceAllowed                                                                                                       bool   `json:"source_proof_janus_surface_allowed"`
	SourceProofCoocLearningAllowed                                                                                                       bool   `json:"source_proof_cooc_learning_allowed"`
	SourceProofDeltaHarvestAllowed                                                                                                       bool   `json:"source_proof_delta_harvest_allowed"`
	SourceProofBodyMutationAllowed                                                                                                       bool   `json:"source_proof_body_mutation_allowed"`
	SourceProofRollbackRequired                                                                                                          bool   `json:"source_proof_rollback_required"`
	SourceProofReadOnly                                                                                                                  bool   `json:"source_proof_read_only"`
	SourceProofReplayOnly                                                                                                                bool   `json:"source_proof_replay_only"`
	SourceProofAuthorityGranted                                                                                                          bool   `json:"source_proof_authority_granted"`
	SourceProofContractsReady                                                                                                            bool   `json:"source_proof_contracts_ready"`
	SourceProofWriteAllowed                                                                                                              bool   `json:"source_proof_write_allowed"`
	SourceProofAdmissionAllowed                                                                                                          bool   `json:"source_proof_admission_allowed"`
	SourceProofLiveAdmissionEnabled                                                                                                      bool   `json:"source_proof_live_admission_enabled"`
	SourceProofMutatesState                                                                                                              bool   `json:"source_proof_mutates_state"`
	SourceProofBodyTarget                                                                                                                string `json:"source_proof_body_target"`
	SourceProofPassed                                                                                                                    bool   `json:"source_proof_passed"`
	SourceProofReason                                                                                                                    string `json:"source_proof_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID                              string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady                           bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID                                   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady                                bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID                                            string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady                                         bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                                                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID                                                string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady                                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID                                                         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady                                                      bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                                                                 string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                                                              bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                                                                    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                                                                 bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                                                                   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                                                              bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                                                            bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                                                                         bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWriterInventoryVerified                                                                                                        bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                                                                        bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                                                              bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                                                                     bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                                                                     bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                                                                 bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                                                               bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                                                               bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                                                                       bool   `json:"source_read_only"`
	SourceReplayOnly                                                                                                                     bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_PRECONDITION_DECISION_REPORT")
	}
	preconditionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision output path missing")
	}
	precondition, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportForAssert(preconditionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReportError(precondition, root); err != nil {
		return err
	}
	decision := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport{
		Schema:         admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema,
		Status:         "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run",
		Target:         "live_route_admission_next_step",
		TargetKind:     "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision",
		TargetMode:     "closed_decision_receipt_dry_run",
		Action:         "decide_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_dry_run",
		Decision:       "shadow_ready",
		LedgerState:    "blocked",
		LedgerAction:   "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ledger_append",
		LedgerContract: "none", LedgerEntrypoint: "none", LedgerReceiptShape: "none", LedgerWriteScope: "none",
		LedgerReady: false, LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionConsumed:      true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionRequired:      true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision: true,
		ReceiptShape:                 "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_receipt",
		DecisionKind:                 "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision",
		DecisionMode:                 "closed_proof_precondition_decision",
		DecisionStage:                "post_preflight_gate_candidate_store_reader_proof_precondition_pre_live_admission_decision",
		ProofPreconditionVerified:    true,
		PreconditionHashVerified:     true,
		PreconditionReadBackVerified: true,
		ProofVerified:                precondition.ProofVerified,
		ProofHashVerified:            precondition.ProofHashVerified,
		ProofReadBackVerified:        precondition.ProofReadBackVerified,
		StoreReaderVerified:          precondition.StoreReaderVerified,
		StoreVerified:                precondition.StoreVerified,
		CandidateVerified:            precondition.CandidateVerified,
		GateVerified:                 precondition.GateVerified,
		PreflightVerified:            precondition.PreflightVerified,
		BoundaryVerified:             precondition.BoundaryVerified,
		ObservationVerified:          precondition.ObservationVerified,
		ReceiverVerified:             precondition.ReceiverVerified,
		IntentVerified:               precondition.IntentVerified,
		FinalGateVerified:            precondition.FinalGateVerified,
		SealVerified:                 precondition.SealVerified,
		PermitVerified:               precondition.PermitVerified,
		AuthorityVerified:            precondition.AuthorityVerified,
		ReaderHashVerified:           precondition.ReaderHashVerified,
		ReaderReplayVerified:         precondition.ReaderReplayVerified,
		ReaderReadBackVerified:       precondition.ReaderReadBackVerified,
		StoreHashVerified:            precondition.StoreHashVerified,
		StoreReadBackVerified:        precondition.StoreReadBackVerified,
		AdmissionRequired:            true,
		ShadowOnly:                   true,
		GraftAllowed:                 false,
		DryRunOnly:                   true,
		LiveReady:                    precondition.LiveReady,
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
		Reason:                       "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition accepted as closed shadow-ready decision receipt",

		SourceSchema: precondition.Schema, SourceStatus: precondition.Status, SourceTarget: precondition.Target, SourceReport: preconditionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID:       precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady:    precondition.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID: precondition.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash:     precondition.PreconditionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack: precondition.ReadBackHash,
		SourcePreconditionAction: precondition.Action, SourcePreconditionReceiptShape: precondition.ReceiptShape, SourcePreconditionKind: precondition.PreconditionKind, SourcePreconditionMode: precondition.PreconditionMode, SourcePreconditionStage: precondition.PreconditionStage,
		SourcePreconditionLedgerReady: precondition.LedgerReady, SourcePreconditionLedgerAppendAllowed: precondition.LedgerAppendAllowed,
		SourcePreconditionAdmissionRequired: precondition.AdmissionRequired, SourcePreconditionShadowOnly: precondition.ShadowOnly, SourcePreconditionGraftAllowed: precondition.GraftAllowed, SourcePreconditionDryRunOnly: precondition.DryRunOnly, SourcePreconditionLiveReady: precondition.LiveReady,
		SourcePreconditionRawDreamTextAllowed: precondition.RawDreamTextAllowed, SourcePreconditionRawDreamTextObserved: precondition.RawDreamTextObserved, SourcePreconditionRawDreamTextForwarded: precondition.RawDreamTextForwarded, SourcePreconditionJanusSurfaceAllowed: precondition.JanusSurfaceAllowed, SourcePreconditionCoocLearningAllowed: precondition.CoocLearningAllowed, SourcePreconditionDeltaHarvestAllowed: precondition.DeltaHarvestAllowed, SourcePreconditionBodyMutationAllowed: precondition.BodyMutationAllowed,
		SourcePreconditionRollbackRequired: precondition.RollbackRequired, SourcePreconditionReadOnly: precondition.ReadOnly, SourcePreconditionReplayOnly: precondition.ReplayOnly, SourcePreconditionAuthorityGranted: precondition.AuthorityGranted, SourcePreconditionContractsReady: precondition.ContractsReady, SourcePreconditionWriteAllowed: precondition.WriteAllowed, SourcePreconditionAdmissionAllowed: precondition.AdmissionAllowed, SourcePreconditionLiveAdmissionEnabled: precondition.LiveAdmissionEnabled, SourcePreconditionMutatesState: precondition.MutatesState, SourcePreconditionBodyTarget: precondition.BodyTarget, SourcePreconditionPassed: precondition.Passed, SourcePreconditionReason: precondition.Reason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady:    precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID: precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash:     precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack: precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBack,
		SourceProofAction: precondition.SourceProofAction, SourceProofReceiptShape: precondition.SourceProofReceiptShape, SourceProofKind: precondition.SourceProofKind, SourceProofMode: precondition.SourceProofMode, SourceProofStage: precondition.SourceProofStage,
		SourceProofLedgerReady: precondition.SourceProofLedgerReady, SourceProofLedgerAppendAllowed: precondition.SourceProofLedgerAppendAllowed,
		SourceProofAdmissionRequired: precondition.SourceProofAdmissionRequired, SourceProofShadowOnly: precondition.SourceProofShadowOnly, SourceProofGraftAllowed: precondition.SourceProofGraftAllowed, SourceProofDryRunOnly: precondition.SourceProofDryRunOnly, SourceProofLiveReady: precondition.SourceProofLiveReady,
		SourceProofRawDreamTextAllowed: precondition.SourceProofRawDreamTextAllowed, SourceProofRawDreamTextObserved: precondition.SourceProofRawDreamTextObserved, SourceProofRawDreamTextForwarded: precondition.SourceProofRawDreamTextForwarded, SourceProofJanusSurfaceAllowed: precondition.SourceProofJanusSurfaceAllowed, SourceProofCoocLearningAllowed: precondition.SourceProofCoocLearningAllowed, SourceProofDeltaHarvestAllowed: precondition.SourceProofDeltaHarvestAllowed, SourceProofBodyMutationAllowed: precondition.SourceProofBodyMutationAllowed,
		SourceProofRollbackRequired: precondition.SourceProofRollbackRequired, SourceProofReadOnly: precondition.SourceProofReadOnly, SourceProofReplayOnly: precondition.SourceProofReplayOnly, SourceProofAuthorityGranted: precondition.SourceProofAuthorityGranted, SourceProofContractsReady: precondition.SourceProofContractsReady, SourceProofWriteAllowed: precondition.SourceProofWriteAllowed, SourceProofAdmissionAllowed: precondition.SourceProofAdmissionAllowed, SourceProofLiveAdmissionEnabled: precondition.SourceProofLiveAdmissionEnabled, SourceProofMutatesState: precondition.SourceProofMutatesState, SourceProofBodyTarget: precondition.SourceProofBodyTarget, SourceProofPassed: precondition.SourceProofPassed, SourceProofReason: precondition.SourceProofReason,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:    precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady: precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:          precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:               precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:            precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:                        precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                     precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                                    precondition.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                            precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:                         precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:                                     precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:                                  precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                                             precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                                          precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                                                precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                                             precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                                               precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                                                     precondition.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                                          precondition.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                                                     precondition.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                                        precondition.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                                                     precondition.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified: precondition.SourceWriterInventoryVerified, SourceWriterPreflightVerified: precondition.SourceWriterPreflightVerified, SourceAdmissionRequired: precondition.SourceAdmissionRequired, SourceShadowOnly: precondition.SourceShadowOnly, SourceDryRunOnly: precondition.SourceDryRunOnly, SourceRequiresWriter: precondition.SourceRequiresWriter, SourceRollbackRequired: precondition.SourceRollbackRequired, SourceRequiresRollback: precondition.SourceRequiresRollback, SourceReadOnly: precondition.SourceReadOnly, SourceReplayOnly: precondition.SourceReplayOnly,
	}
	decision.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID(decision)
	decision.DecisionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash(decision)
	decision.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBackHash(decision)
	decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID(decision)
	if decision.CausalID == "" || decision.DecisionHash == "" || decision.ReadBackHash == "" || decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID == "" || decision.DecisionHash == decision.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision read-back proof failed")
	}
	raw, err := json.MarshalIndent(decision, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_report=%s\n", outputPath, preconditionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" || report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" || report.TargetMode != "closed_decision_receipt_dry_run" || report.Action != "decide_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision route shape mismatch")
	}
	if report.Decision != "shadow_ready" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision decision mismatch: got %q want %q", report.Decision, "shadow_ready")
	}
	if report.LedgerState != "blocked" || report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ledger_append" || report.LedgerContract != "none" || report.LedgerEntrypoint != "none" || report.LedgerReceiptShape != "none" || report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_receipt" || report.DecisionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision" || report.DecisionMode != "closed_proof_precondition_decision" || report.DecisionStage != "post_preflight_gate_candidate_store_reader_proof_precondition_pre_live_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision},
		{"proof_precondition_verified", report.ProofPreconditionVerified}, {"precondition_hash_verified", report.PreconditionHashVerified}, {"precondition_read_back_verified", report.PreconditionReadBackVerified},
		{"proof_verified", report.ProofVerified}, {"proof_hash_verified", report.ProofHashVerified}, {"proof_read_back_verified", report.ProofReadBackVerified},
		{"store_reader_verified", report.StoreReaderVerified}, {"store_verified", report.StoreVerified}, {"candidate_verified", report.CandidateVerified}, {"gate_verified", report.GateVerified}, {"preflight_verified", report.PreflightVerified}, {"boundary_verified", report.BoundaryVerified}, {"observation_verified", report.ObservationVerified}, {"receiver_verified", report.ReceiverVerified}, {"intent_verified", report.IntentVerified}, {"final_gate_verified", report.FinalGateVerified}, {"seal_verified", report.SealVerified}, {"permit_verified", report.PermitVerified}, {"authority_verified", report.AuthorityVerified}, {"reader_hash_verified", report.ReaderHashVerified}, {"reader_replay_verified", report.ReaderReplayVerified}, {"reader_read_back_verified", report.ReaderReadBackVerified}, {"store_hash_verified", report.StoreHashVerified}, {"store_read_back_verified", report.StoreReadBackVerified},
		{"admission_required", report.AdmissionRequired}, {"shadow_only", report.ShadowOnly}, {"dry_run_only", report.DryRunOnly}, {"live_ready", report.LiveReady}, {"rollback_required", report.RollbackRequired}, {"read_only", report.ReadOnly}, {"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReady},
		{"source_precondition_admission_required", report.SourcePreconditionAdmissionRequired}, {"source_precondition_shadow_only", report.SourcePreconditionShadowOnly}, {"source_precondition_dry_run_only", report.SourcePreconditionDryRunOnly}, {"source_precondition_live_ready", report.SourcePreconditionLiveReady}, {"source_precondition_rollback_required", report.SourcePreconditionRollbackRequired}, {"source_precondition_read_only", report.SourcePreconditionReadOnly}, {"source_precondition_replay_only", report.SourcePreconditionReplayOnly}, {"source_precondition_passed", report.SourcePreconditionPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady}, {"source_proof_admission_required", report.SourceProofAdmissionRequired}, {"source_proof_shadow_only", report.SourceProofShadowOnly}, {"source_proof_dry_run_only", report.SourceProofDryRunOnly}, {"source_proof_live_ready", report.SourceProofLiveReady}, {"source_proof_rollback_required", report.SourceProofRollbackRequired}, {"source_proof_read_only", report.SourceProofReadOnly}, {"source_proof_replay_only", report.SourceProofReplayOnly}, {"source_proof_passed", report.SourceProofPassed},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady}, {"source_weighted_admission_resonance_graft_admission_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady}, {"source_weighted_admission_resonance_graft_admission_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady}, {"source_weighted_admission_resonance_graft_admission_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady}, {"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady}, {"source_weighted_admission_resonance_graft_admission_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady}, {"source_writer_inventory_verified", report.SourceWriterInventoryVerified}, {"source_writer_preflight_verified", report.SourceWriterPreflightVerified}, {"source_admission_required", report.SourceAdmissionRequired}, {"source_shadow_only", report.SourceShadowOnly}, {"source_dry_run_only", report.SourceDryRunOnly}, {"source_requires_writer", report.SourceRequiresWriter}, {"source_rollback_required", report.SourceRollbackRequired}, {"source_requires_rollback", report.SourceRequiresRollback}, {"source_read_only", report.SourceReadOnly}, {"source_replay_only", report.SourceReplayOnly}, {"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady}, {"ledger_append_allowed", report.LedgerAppendAllowed}, {"graft_allowed", report.GraftAllowed}, {"raw_dream_text_allowed", report.RawDreamTextAllowed}, {"raw_dream_text_observed", report.RawDreamTextObserved}, {"raw_dream_text_forwarded", report.RawDreamTextForwarded}, {"janus_surface_allowed", report.JanusSurfaceAllowed}, {"cooc_learning_allowed", report.CoocLearningAllowed}, {"delta_harvest_allowed", report.DeltaHarvestAllowed}, {"body_mutation_allowed", report.BodyMutationAllowed}, {"authority_granted", report.AuthorityGranted}, {"contracts_ready", report.ContractsReady}, {"write_allowed", report.WriteAllowed}, {"admission_allowed", report.AdmissionAllowed}, {"live_admission_enabled", report.LiveAdmissionEnabled}, {"mutates_state", report.MutatesState},
		{"source_precondition_ledger_ready", report.SourcePreconditionLedgerReady}, {"source_precondition_ledger_append_allowed", report.SourcePreconditionLedgerAppendAllowed}, {"source_precondition_graft_allowed", report.SourcePreconditionGraftAllowed}, {"source_precondition_raw_dream_text_allowed", report.SourcePreconditionRawDreamTextAllowed}, {"source_precondition_raw_dream_text_observed", report.SourcePreconditionRawDreamTextObserved}, {"source_precondition_raw_dream_text_forwarded", report.SourcePreconditionRawDreamTextForwarded}, {"source_precondition_janus_surface_allowed", report.SourcePreconditionJanusSurfaceAllowed}, {"source_precondition_cooc_learning_allowed", report.SourcePreconditionCoocLearningAllowed}, {"source_precondition_delta_harvest_allowed", report.SourcePreconditionDeltaHarvestAllowed}, {"source_precondition_body_mutation_allowed", report.SourcePreconditionBodyMutationAllowed}, {"source_precondition_authority_granted", report.SourcePreconditionAuthorityGranted}, {"source_precondition_contracts_ready", report.SourcePreconditionContractsReady}, {"source_precondition_write_allowed", report.SourcePreconditionWriteAllowed}, {"source_precondition_admission_allowed", report.SourcePreconditionAdmissionAllowed}, {"source_precondition_live_admission_enabled", report.SourcePreconditionLiveAdmissionEnabled}, {"source_precondition_mutates_state", report.SourcePreconditionMutatesState},
		{"source_proof_ledger_ready", report.SourceProofLedgerReady}, {"source_proof_ledger_append_allowed", report.SourceProofLedgerAppendAllowed}, {"source_proof_graft_allowed", report.SourceProofGraftAllowed}, {"source_proof_raw_dream_text_allowed", report.SourceProofRawDreamTextAllowed}, {"source_proof_raw_dream_text_observed", report.SourceProofRawDreamTextObserved}, {"source_proof_raw_dream_text_forwarded", report.SourceProofRawDreamTextForwarded}, {"source_proof_janus_surface_allowed", report.SourceProofJanusSurfaceAllowed}, {"source_proof_cooc_learning_allowed", report.SourceProofCoocLearningAllowed}, {"source_proof_delta_harvest_allowed", report.SourceProofDeltaHarvestAllowed}, {"source_proof_body_mutation_allowed", report.SourceProofBodyMutationAllowed}, {"source_proof_authority_granted", report.SourceProofAuthorityGranted}, {"source_proof_contracts_ready", report.SourceProofContractsReady}, {"source_proof_write_allowed", report.SourceProofWriteAllowed}, {"source_proof_admission_allowed", report.SourceProofAdmissionAllowed}, {"source_proof_live_admission_enabled", report.SourceProofLiveAdmissionEnabled}, {"source_proof_mutates_state", report.SourceProofMutatesState}, {"source_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct{ name, value string }{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID}, {"causal_id", report.CausalID}, {"decision_hash", report.DecisionHash}, {"read_back_hash", report.ReadBackHash}, {"source_report", report.SourceReport}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionCausalID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID}, {"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_satisfied_dry_run" || report.SourceTarget != "live_route_admission_next_step" || report.SourcePreconditionAction != "consume_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_before_live_route_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source precondition route mismatch")
	}
	if report.SourcePreconditionReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_receipt" || report.SourcePreconditionKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition" || report.SourcePreconditionMode != "closed_receipt_consumption" || report.SourcePreconditionStage != "post_preflight_gate_candidate_store_reader_proof_pre_live_admission_precondition" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source precondition shape mismatch")
	}
	if report.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" || report.SourceProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" || report.SourceProofMode != "closed_read_back_reader_proof" || report.SourceProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source proof shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourcePreconditionBodyTarget != "none" || report.SourceProofBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-") || !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-causal-") || !strings.HasPrefix(report.DecisionHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-") || !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-read-") || report.DecisionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") || !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision causal_id mismatch")
	}
	if report.DecisionHash == "" || report.DecisionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision decision_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision read_back_hash mismatch")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof precondition accepted as closed shadow-ready decision receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionCausalID(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport) string {
	h := hashJSON(struct{ SourcePreconditionID, SourcePreconditionReadBack, SourceProofID, SourceReaderID, SourceStoreID, SourceCandidateID, SourceGateID, Target, DecisionKind, DecisionStage string }{
		decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, decision.Target, decision.DecisionKind, decision.DecisionStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionHash(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport) string {
	h := hashJSON(struct {
		CausalID, SourcePreconditionID, SourcePreconditionHash, SourcePreconditionReadBack, Decision, DecisionMode, ReceiptShape                                                              string
		ProofPreconditionVerified, PreconditionHashVerified, PreconditionReadBackVerified, ReadOnly, ReplayOnly, AdmissionRequired, ShadowOnly, DryRunOnly, GraftAllowed, LedgerAppendAllowed bool
	}{
		decision.CausalID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionHash, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionReadBack, decision.Decision, decision.DecisionMode, decision.ReceiptShape, decision.ProofPreconditionVerified, decision.PreconditionHashVerified, decision.PreconditionReadBackVerified, decision.ReadOnly, decision.ReplayOnly, decision.AdmissionRequired, decision.ShadowOnly, decision.DryRunOnly, decision.GraftAllowed, decision.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReadBackHash(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport) string {
	h := hashJSON(struct {
		DecisionHash, SourcePreconditionID, SourceProofID, DecisionKind                          string
		DecisionReady, BodyMutation, LiveAdmission, WriteAllowed, AdmissionAllowed, LedgerAppend bool
	}{
		decision.DecisionHash, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, decision.DecisionKind, decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady, decision.BodyMutationAllowed, decision.LiveAdmissionEnabled, decision.WriteAllowed, decision.AdmissionAllowed, decision.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionID(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport) string {
	h := hashJSON(struct {
		Schema, Status, Action, Decision, SourceReport, SourcePreconditionID, SourceProofID, SourceReaderID, SourceStoreID, SourceCandidateID, SourceGateID, CausalID, DecisionHash, ReadBackHash, ReceiptShape, DecisionKind, DecisionMode, DecisionStage, BodyTarget                                                      string
		Ready, ProofPreconditionVerified, PreconditionHashVerified, PreconditionReadBackVerified, AdmissionRequired, ShadowOnly, GraftAllowed, DryRunOnly, ReadOnly, ReplayOnly, LiveReady, ContractsReady, WriteAllowed, AdmissionAllowed, LiveAdmissionEnabled, MutatesState, LedgerAppendAllowed, NextStepBlockedWithout bool
	}{
		decision.Schema, decision.Status, decision.Action, decision.Decision, decision.SourceReport, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, decision.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, decision.CausalID, decision.DecisionHash, decision.ReadBackHash, decision.ReceiptShape, decision.DecisionKind, decision.DecisionMode, decision.DecisionStage, decision.BodyTarget, decision.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReady, decision.ProofPreconditionVerified, decision.PreconditionHashVerified, decision.PreconditionReadBackVerified, decision.AdmissionRequired, decision.ShadowOnly, decision.GraftAllowed, decision.DryRunOnly, decision.ReadOnly, decision.ReplayOnly, decision.LiveReady, decision.ContractsReady, decision.WriteAllowed, decision.AdmissionAllowed, decision.LiveAdmissionEnabled, decision.MutatesState, decision.LedgerAppendAllowed, decision.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecision,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-precondition-decision-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofPreconditionDecisionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof precondition decision decode failed: %w", err)
	}
	return report, root, nil
}
