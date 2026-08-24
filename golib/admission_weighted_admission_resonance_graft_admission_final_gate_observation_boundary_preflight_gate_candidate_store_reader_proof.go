package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport struct {
	Schema                                                                                                          string `json:"schema"`
	Status                                                                                                          string `json:"status"`
	Target                                                                                                          string `json:"target"`
	TargetKind                                                                                                      string `json:"target_kind"`
	TargetMode                                                                                                      string `json:"target_mode"`
	Action                                                                                                          string `json:"action"`
	LedgerState                                                                                                     string `json:"ledger_state"`
	LedgerAction                                                                                                    string `json:"ledger_action"`
	LedgerContract                                                                                                  string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                                string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                              string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                                string `json:"ledger_write_scope"`
	LedgerReady                                                                                                     bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                             bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderConsumed   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderRequired   bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id"`
	ReceiptShape                                                                                                    string `json:"receipt_shape"`
	ProofKind                                                                                                       string `json:"proof_kind"`
	ProofMode                                                                                                       string `json:"proof_mode"`
	ProofStage                                                                                                      string `json:"proof_stage"`
	CausalID                                                                                                        string `json:"causal_id"`
	ProofHash                                                                                                       string `json:"proof_hash"`
	ReadBackHash                                                                                                    string `json:"read_back_hash"`
	StoreReaderVerified                                                                                             bool   `json:"store_reader_verified"`
	StoreVerified                                                                                                   bool   `json:"store_verified"`
	CandidateVerified                                                                                               bool   `json:"candidate_verified"`
	GateVerified                                                                                                    bool   `json:"gate_verified"`
	PreflightVerified                                                                                               bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                                bool   `json:"boundary_verified"`
	ObservationVerified                                                                                             bool   `json:"observation_verified"`
	ReceiverVerified                                                                                                bool   `json:"receiver_verified"`
	IntentVerified                                                                                                  bool   `json:"intent_verified"`
	FinalGateVerified                                                                                               bool   `json:"final_gate_verified"`
	SealVerified                                                                                                    bool   `json:"seal_verified"`
	PermitVerified                                                                                                  bool   `json:"permit_verified"`
	AuthorityVerified                                                                                               bool   `json:"authority_verified"`
	ReaderHashVerified                                                                                              bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                                                            bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                                                                          bool   `json:"reader_read_back_verified"`
	StoreHashVerified                                                                                               bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                           bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                               bool   `json:"admission_required"`
	ShadowOnly                                                                                                      bool   `json:"shadow_only"`
	GraftAllowed                                                                                                    bool   `json:"graft_allowed"`
	DryRunOnly                                                                                                      bool   `json:"dry_run_only"`
	LiveReady                                                                                                       bool   `json:"live_ready"`
	RawDreamTextAllowed                                                                                             bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                            bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                           bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                             bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                             bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                             bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                             bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                                                                bool   `json:"rollback_required"`
	ReadOnly                                                                                                        bool   `json:"read_only"`
	ReplayOnly                                                                                                      bool   `json:"replay_only"`
	AuthorityGranted                                                                                                bool   `json:"authority_granted"`
	ContractsReady                                                                                                  bool   `json:"contracts_ready"`
	WriteAllowed                                                                                                    bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                                bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                            bool   `json:"live_admission_enabled"`
	MutatesState                                                                                                    bool   `json:"mutates_state"`
	BodyTarget                                                                                                      string `json:"body_target"`
	Passed                                                                                                          bool   `json:"passed"`
	Reason                                                                                                          string `json:"reason"`

	SourceSchema                                                                                                            string `json:"source_schema"`
	SourceStatus                                                                                                            string `json:"source_status"`
	SourceTarget                                                                                                            string `json:"source_target"`
	SourceReport                                                                                                            string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID           string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady        bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash         string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash   string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_replay_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_read_back_hash"`
	SourceReaderAction                                                                                                      string `json:"source_reader_action"`
	SourceReaderReceiptShape                                                                                                string `json:"source_reader_receipt_shape"`
	SourceReaderKind                                                                                                        string `json:"source_reader_kind"`
	SourceReaderMode                                                                                                        string `json:"source_reader_mode"`
	SourceReaderStage                                                                                                       string `json:"source_reader_stage"`
	SourceReaderReadOnly                                                                                                    bool   `json:"source_reader_read_only"`
	SourceReaderReplayOnly                                                                                                  bool   `json:"source_reader_replay_only"`
	SourceReaderStoreVerified                                                                                               bool   `json:"source_reader_store_verified"`
	SourceReaderCandidateVerified                                                                                           bool   `json:"source_reader_candidate_verified"`
	SourceReaderGateVerified                                                                                                bool   `json:"source_reader_gate_verified"`
	SourceReaderPreflightVerified                                                                                           bool   `json:"source_reader_preflight_verified"`
	SourceReaderBoundaryVerified                                                                                            bool   `json:"source_reader_boundary_verified"`
	SourceReaderObservationVerified                                                                                         bool   `json:"source_reader_observation_verified"`
	SourceReaderFinalGateVerified                                                                                           bool   `json:"source_reader_final_gate_verified"`
	SourceReaderSealVerified                                                                                                bool   `json:"source_reader_seal_verified"`
	SourceReaderPermitVerified                                                                                              bool   `json:"source_reader_permit_verified"`
	SourceReaderAuthorityVerified                                                                                           bool   `json:"source_reader_authority_verified"`
	SourceReaderStoreHashVerified                                                                                           bool   `json:"source_reader_store_hash_verified"`
	SourceReaderStoreReadBackVerified                                                                                       bool   `json:"source_reader_store_read_back_verified"`
	SourceReaderAdmissionRequired                                                                                           bool   `json:"source_reader_admission_required"`
	SourceReaderShadowOnly                                                                                                  bool   `json:"source_reader_shadow_only"`
	SourceReaderDryRunOnly                                                                                                  bool   `json:"source_reader_dry_run_only"`
	SourceReaderLiveReady                                                                                                   bool   `json:"source_reader_live_ready"`
	SourceReaderRollbackRequired                                                                                            bool   `json:"source_reader_rollback_required"`
	SourceReaderLedgerReady                                                                                                 bool   `json:"source_reader_ledger_ready"`
	SourceReaderLedgerAppendAllowed                                                                                         bool   `json:"source_reader_ledger_append_allowed"`
	SourceReaderRawDreamTextAllowed                                                                                         bool   `json:"source_reader_raw_dream_text_allowed"`
	SourceReaderRawDreamTextObserved                                                                                        bool   `json:"source_reader_raw_dream_text_observed"`
	SourceReaderRawDreamTextForwarded                                                                                       bool   `json:"source_reader_raw_dream_text_forwarded"`
	SourceReaderJanusSurfaceAllowed                                                                                         bool   `json:"source_reader_janus_surface_allowed"`
	SourceReaderCoocLearningAllowed                                                                                         bool   `json:"source_reader_cooc_learning_allowed"`
	SourceReaderDeltaHarvestAllowed                                                                                         bool   `json:"source_reader_delta_harvest_allowed"`
	SourceReaderBodyMutationAllowed                                                                                         bool   `json:"source_reader_body_mutation_allowed"`
	SourceReaderAuthorityGranted                                                                                            bool   `json:"source_reader_authority_granted"`
	SourceReaderContractsReady                                                                                              bool   `json:"source_reader_contracts_ready"`
	SourceReaderWriteAllowed                                                                                                bool   `json:"source_reader_write_allowed"`
	SourceReaderAdmissionAllowed                                                                                            bool   `json:"source_reader_admission_allowed"`
	SourceReaderLiveAdmissionEnabled                                                                                        bool   `json:"source_reader_live_admission_enabled"`
	SourceReaderMutatesState                                                                                                bool   `json:"source_reader_mutates_state"`
	SourceReaderBodyTarget                                                                                                  string `json:"source_reader_body_target"`
	SourceReaderPassed                                                                                                      bool   `json:"source_reader_passed"`
	SourceReaderReason                                                                                                      string `json:"source_reader_reason"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash"`
	SourceStoreReceiptShape                                                                                       string `json:"source_store_receipt_shape"`
	SourceStoreKind                                                                                               string `json:"source_store_kind"`
	SourceStoreMode                                                                                               string `json:"source_store_mode"`
	SourceStoreStage                                                                                              string `json:"source_store_stage"`
	SourceStoreAppendOnly                                                                                         bool   `json:"source_store_append_only"`
	SourceStoreReadBack                                                                                           bool   `json:"source_store_read_back"`
	SourceStoreReceiptPersisted                                                                                   bool   `json:"source_store_receipt_persisted"`
	SourceStoreReceiptVerified                                                                                    bool   `json:"source_store_receipt_verified"`
	SourceStoreLedgerReady                                                                                        bool   `json:"source_store_ledger_ready"`
	SourceStoreLedgerAppendAllowed                                                                                bool   `json:"source_store_ledger_append_allowed"`
	SourceStoreRawDreamTextAllowed                                                                                bool   `json:"source_store_raw_dream_text_allowed"`
	SourceStoreRawDreamTextObserved                                                                               bool   `json:"source_store_raw_dream_text_observed"`
	SourceStoreRawDreamTextForwarded                                                                              bool   `json:"source_store_raw_dream_text_forwarded"`
	SourceStoreJanusSurfaceAllowed                                                                                bool   `json:"source_store_janus_surface_allowed"`
	SourceStoreCoocLearningAllowed                                                                                bool   `json:"source_store_cooc_learning_allowed"`
	SourceStoreDeltaHarvestAllowed                                                                                bool   `json:"source_store_delta_harvest_allowed"`
	SourceStoreBodyMutationAllowed                                                                                bool   `json:"source_store_body_mutation_allowed"`
	SourceStoreAuthorityGranted                                                                                   bool   `json:"source_store_authority_granted"`
	SourceStoreContractsReady                                                                                     bool   `json:"source_store_contracts_ready"`
	SourceStoreWriteAllowed                                                                                       bool   `json:"source_store_write_allowed"`
	SourceStoreAdmissionAllowed                                                                                   bool   `json:"source_store_admission_allowed"`
	SourceStoreLiveAdmissionEnabled                                                                               bool   `json:"source_store_live_admission_enabled"`
	SourceStoreMutatesState                                                                                       bool   `json:"source_store_mutates_state"`
	SourceStoreBodyTarget                                                                                         string `json:"source_store_body_target"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash"`
	SourceCandidateReceiptShape                                                                              string `json:"source_candidate_receipt_shape"`
	SourceCandidateState                                                                                     string `json:"source_candidate_state"`
	SourceCandidateKind                                                                                      string `json:"source_candidate_kind"`
	SourceCandidateMode                                                                                      string `json:"source_candidate_mode"`
	SourceCandidateStage                                                                                     string `json:"source_candidate_stage"`
	SourceCandidateDryRunOnly                                                                                bool   `json:"source_candidate_dry_run_only"`
	SourceCandidateGateVerified                                                                              bool   `json:"source_candidate_gate_verified"`
	SourceCandidatePreflightVerified                                                                         bool   `json:"source_candidate_preflight_verified"`
	SourceCandidateBoundaryVerified                                                                          bool   `json:"source_candidate_boundary_verified"`
	SourceCandidateObservationVerified                                                                       bool   `json:"source_candidate_observation_verified"`
	SourceCandidateReadBackVerified                                                                          bool   `json:"source_candidate_read_back_verified"`
	SourceCandidateOpened                                                                                    bool   `json:"source_candidate_opened"`
	SourceCandidateRawDreamTextObserved                                                                      bool   `json:"source_candidate_raw_dream_text_observed"`
	SourceCandidateRawDreamTextForwarded                                                                     bool   `json:"source_candidate_raw_dream_text_forwarded"`
	SourceCandidateRawDreamTextAllowed                                                                       bool   `json:"source_candidate_raw_dream_text_allowed"`
	SourceCandidateBodyMutationAllowed                                                                       bool   `json:"source_candidate_body_mutation_allowed"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateHash                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReady                                   bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_ready"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly                              bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved                             bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded                            bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed                              bool   `json:"source_final_gate_observation_boundary_preflight_gate_body_mutation_allowed"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID    string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID             string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady          bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID                     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady                  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID                        string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady                     bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady                       bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady                                  bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady                                bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady                             bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWriterInventoryVerified                                                            bool   `json:"source_writer_inventory_verified"`
	SourceWriterPreflightVerified                                                            bool   `json:"source_writer_preflight_verified"`
	SourceAdmissionRequired                                                                  bool   `json:"source_admission_required"`
	SourceShadowOnly                                                                         bool   `json:"source_shadow_only"`
	SourceDryRunOnly                                                                         bool   `json:"source_dry_run_only"`
	SourceRequiresWriter                                                                     bool   `json:"source_requires_writer"`
	SourceRollbackRequired                                                                   bool   `json:"source_rollback_required"`
	SourceRequiresRollback                                                                   bool   `json:"source_requires_rollback"`
	SourceReadOnly                                                                           bool   `json:"source_read_only"`
	SourceReplayOnly                                                                         bool   `json:"source_replay_only"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_PROOF_REPORT")
	}
	readerPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof output path missing")
	}
	reader, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportForAssert(readerPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportError(reader, root); err != nil {
		return err
	}
	proof := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema,
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
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderConsumed:   true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderRequired:   true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProof: true,
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
		ReceiverVerified:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		IntentVerified:         reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
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
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID:           reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady:        reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID:     reader.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash:         reader.ReaderHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash:   reader.ReplayHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash: reader.ReadBackHash,
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

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash:                                    reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash:                            reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
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

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash:                                    reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash:                            reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
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

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:                                    reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash:                            reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                   reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:                              reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified:                        reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved:                             reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:                            reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:                              reader.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed:                              reader.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:             reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:          reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                  reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                        reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                     reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                             reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                  reader.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                             reader.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                reader.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                             reader.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified:                                                            reader.SourceWriterInventoryVerified,
		SourceWriterPreflightVerified:                                                            reader.SourceWriterPreflightVerified,
		SourceAdmissionRequired:                                                                  reader.SourceAdmissionRequired,
		SourceShadowOnly:                                                                         reader.SourceShadowOnly,
		SourceDryRunOnly:                                                                         reader.SourceDryRunOnly,
		SourceRequiresWriter:                                                                     reader.SourceRequiresWriter,
		SourceRollbackRequired:                                                                   reader.SourceRollbackRequired,
		SourceRequiresRollback:                                                                   reader.SourceRequiresRollback,
		SourceReadOnly:                                                                           reader.SourceReadOnly,
		SourceReplayOnly:                                                                         reader.SourceReplayOnly,
	}
	proof.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(proof)
	proof.ProofHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash(proof)
	proof.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(proof)
	proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID(proof)
	if proof.CausalID == "" ||
		proof.ProofHash == "" ||
		proof.ReadBackHash == "" ||
		proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID == "" ||
		proof.ProofHash == proof.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof read-back proof failed")
	}
	raw, err := json.MarshalIndent(proof, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_report=%s\n", outputPath, readerPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" ||
		report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		report.TargetMode != "receipt_only_closed_reader_proof_dry_run" ||
		report.Action != "prove_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof route shape mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ledger_append" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_receipt" ||
		report.ProofKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof" ||
		report.ProofMode != "closed_read_back_reader_proof" ||
		report.ProofStage != "post_preflight_gate_candidate_store_reader_pre_live_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderRequired},
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
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady},
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
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady},
		{"source_store_append_only", report.SourceStoreAppendOnly},
		{"source_store_read_back", report.SourceStoreReadBack},
		{"source_store_receipt_persisted", report.SourceStoreReceiptPersisted},
		{"source_store_receipt_verified", report.SourceStoreReceiptVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady},
		{"source_candidate_dry_run_only", report.SourceCandidateDryRunOnly},
		{"source_candidate_gate_verified", report.SourceCandidateGateVerified},
		{"source_candidate_preflight_verified", report.SourceCandidatePreflightVerified},
		{"source_candidate_boundary_verified", report.SourceCandidateBoundaryVerified},
		{"source_candidate_observation_verified", report.SourceCandidateObservationVerified},
		{"source_candidate_read_back_verified", report.SourceCandidateReadBackVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID},
		{"causal_id", report.CausalID},
		{"proof_hash", report.ProofHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_replay_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID},
		{"source_admission_final_gate_observation_boundary_preflight_gate_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash},
		{"source_admission_final_gate_observation_boundary_preflight_gate_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run" ||
		report.SourceTarget != "live_route_admission_next_step" ||
		report.SourceReaderAction != "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source reader route mismatch")
	}
	if report.SourceReaderReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt" ||
		report.SourceReaderKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		report.SourceReaderMode != "read_only_replay" ||
		report.SourceReaderStage != "post_preflight_gate_candidate_store_pre_live_admission_reader" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source reader shape mismatch")
	}
	if report.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt" ||
		report.SourceStoreKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		report.SourceStoreMode != "append_only_read_back_store" ||
		report.SourceStoreStage != "post_preflight_gate_candidate_pre_live_admission_store" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source store shape mismatch")
	}
	if report.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_receipt" ||
		report.SourceCandidateState != "blocked" ||
		report.SourceCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.SourceCandidateMode != "no_mutation_preflight_gate_candidate" ||
		report.SourceCandidateStage != "post_preflight_gate_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source candidate shape mismatch")
	}
	if report.BodyTarget != "none" || report.SourceReaderBodyTarget != "none" || report.SourceStoreBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof body target mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-") ||
		!strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-causal-") ||
		!strings.HasPrefix(report.ProofHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-") ||
		!strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-read-") ||
		report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-replay-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source reader proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source store proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID, "weighted-resonance-graft-admission-final-gate-observation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-receiver-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof source chain prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof causal_id mismatch")
	}
	if report.ProofHash == "" || report.ProofHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof proof_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof read_back_hash mismatch")
	}
	if report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID == "" ||
		report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store reader proof sealed without ledger append or body mutation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofCausalID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
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
		SourceReaderID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceReaderReadBack: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash,
		SourceStoreID:        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceStoreReadBack:  proof.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		SourceCandidateID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceGateID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourcePreflightID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceBoundaryID:     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceObservationID:  proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		Target:               proof.Target,
		ProofKind:            proof.ProofKind,
		ProofStage:           proof.ProofStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
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
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceReaderHash:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash,
		SourceReaderReplayHash: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash,
		SourceReaderReadBack:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
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
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReadBackHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
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
		SourceReaderID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidate:     proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		ProofKind:           proof.ProofKind,
		ProofReady:          proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
		BodyMutation:        proof.BodyMutationAllowed,
		LiveAdmission:       proof.LiveAdmissionEnabled,
		WriteAllowed:        proof.WriteAllowed,
		AdmissionAllowed:    proof.AdmissionAllowed,
		LedgerAppendAllowed: proof.LedgerAppendAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport) string {
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
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceGateID:           proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourcePreflightID:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceBoundaryID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceObservationID:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceReceiverID:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		CausalID:               proof.CausalID,
		ProofHash:              proof.ProofHash,
		ReadBackHash:           proof.ReadBackHash,
		Ready:                  proof.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReady,
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
		SourceReaderReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		SourceStoreReady:       proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceCandidateReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceGateReady:        proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourcePreflightReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceBoundaryReady:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceObservationReady: proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceReceiverReady:    proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceIntentReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceFinalGateReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceSealReady:        proof.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceAuthorityReady:   proof.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourcePermitReady:      proof.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-proof-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderProofReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof decode failed: %w", err)
	}
	return report, root, nil
}
