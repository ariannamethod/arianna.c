package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_proof.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport struct {
	Schema                                                              string `json:"schema"`
	Status                                                              string `json:"status"`
	Target                                                              string `json:"target"`
	TargetKind                                                          string `json:"target_kind"`
	TargetMode                                                          string `json:"target_mode"`
	Action                                                              string `json:"action"`
	WeightedAdmissionResonanceGraftAdmissionProofReady                  bool   `json:"weighted_admission_resonance_graft_admission_proof_ready"`
	WeightedAdmissionResonanceGraftCandidateStoreReaderConsumed         bool   `json:"weighted_admission_resonance_graft_candidate_store_reader_consumed"`
	WeightedAdmissionResonanceGraftCandidateStoreReaderRequired         bool   `json:"weighted_admission_resonance_graft_candidate_store_reader_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionProof                  bool   `json:"next_step_blocked_without_resonance_graft_admission_proof"`
	WeightedAdmissionResonanceGraftAdmissionProofID                     string `json:"weighted_admission_resonance_graft_admission_proof_id"`
	ReceiptShape                                                        string `json:"receipt_shape"`
	ProofKind                                                           string `json:"proof_kind"`
	ProofMode                                                           string `json:"proof_mode"`
	ProofStage                                                          string `json:"proof_stage"`
	CausalID                                                            string `json:"causal_id"`
	ProofHash                                                           string `json:"proof_hash"`
	ReadBackHash                                                        string `json:"read_back_hash"`
	StoreReaderVerified                                                 bool   `json:"store_reader_verified"`
	StoreVerified                                                       bool   `json:"store_verified"`
	CandidateVerified                                                   bool   `json:"candidate_verified"`
	GateVerified                                                        bool   `json:"gate_verified"`
	PreflightVerified                                                   bool   `json:"preflight_verified"`
	BoundaryVerified                                                    bool   `json:"boundary_verified"`
	ObservationVerified                                                 bool   `json:"observation_verified"`
	ReceiverVerified                                                    bool   `json:"receiver_verified"`
	IntentVerified                                                      bool   `json:"intent_verified"`
	FinalGateVerified                                                   bool   `json:"final_gate_verified"`
	SealVerified                                                        bool   `json:"seal_verified"`
	PermitVerified                                                      bool   `json:"permit_verified"`
	AuthorityVerified                                                   bool   `json:"authority_verified"`
	ReaderHashVerified                                                  bool   `json:"reader_hash_verified"`
	ReaderReplayVerified                                                bool   `json:"reader_replay_verified"`
	ReaderReadBackVerified                                              bool   `json:"reader_read_back_verified"`
	AdmissionRequired                                                   bool   `json:"admission_required"`
	ShadowOnly                                                          bool   `json:"shadow_only"`
	GraftAllowed                                                        bool   `json:"graft_allowed"`
	DryRunOnly                                                          bool   `json:"dry_run_only"`
	LiveReady                                                           bool   `json:"live_ready"`
	RawDreamTextAllowed                                                 bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                               bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                 bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                 bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                 bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                 bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                    bool   `json:"rollback_required"`
	ReadOnly                                                            bool   `json:"read_only"`
	ReplayOnly                                                          bool   `json:"replay_only"`
	SourceSchema                                                        string `json:"source_schema"`
	SourceStatus                                                        string `json:"source_status"`
	SourceTarget                                                        string `json:"source_target"`
	SourceReport                                                        string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID         string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady      bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID   string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash       string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_replay_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack   string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_read_back_hash"`
	SourceReaderAction                                                  string `json:"source_reader_action"`
	SourceReaderReceiptShape                                            string `json:"source_reader_receipt_shape"`
	SourceReaderKind                                                    string `json:"source_reader_kind"`
	SourceReaderMode                                                    string `json:"source_reader_mode"`
	SourceReaderStage                                                   string `json:"source_reader_stage"`
	SourceReaderReadOnly                                                bool   `json:"source_reader_read_only"`
	SourceReaderReplayOnly                                              bool   `json:"source_reader_replay_only"`
	SourceReaderStoreVerified                                           bool   `json:"source_reader_store_verified"`
	SourceReaderCandidateVerified                                       bool   `json:"source_reader_candidate_verified"`
	SourceReaderHashVerified                                            bool   `json:"source_reader_hash_verified"`
	SourceReaderReplayVerified                                          bool   `json:"source_reader_replay_verified"`
	SourceReaderReadBackVerified                                        bool   `json:"source_reader_read_back_verified"`
	SourceReaderAdmissionRequired                                       bool   `json:"source_reader_admission_required"`
	SourceReaderShadowOnly                                              bool   `json:"source_reader_shadow_only"`
	SourceReaderGraftAllowed                                            bool   `json:"source_reader_graft_allowed"`
	SourceReaderDryRunOnly                                              bool   `json:"source_reader_dry_run_only"`
	SourceReaderLiveReady                                               bool   `json:"source_reader_live_ready"`
	SourceReaderRawDreamTextAllowed                                     bool   `json:"source_reader_raw_dream_text_allowed"`
	SourceReaderRawDreamTextObserved                                    bool   `json:"source_reader_raw_dream_text_observed"`
	SourceReaderRawDreamTextForwarded                                   bool   `json:"source_reader_raw_dream_text_forwarded"`
	SourceReaderJanusSurfaceAllowed                                     bool   `json:"source_reader_janus_surface_allowed"`
	SourceReaderCoocLearningAllowed                                     bool   `json:"source_reader_cooc_learning_allowed"`
	SourceReaderDeltaHarvestAllowed                                     bool   `json:"source_reader_delta_harvest_allowed"`
	SourceReaderBodyMutationAllowed                                     bool   `json:"source_reader_body_mutation_allowed"`
	SourceReaderRollbackRequired                                        bool   `json:"source_reader_rollback_required"`
	SourceReaderAuthorityGranted                                        bool   `json:"source_reader_authority_granted"`
	SourceReaderContractsReady                                          bool   `json:"source_reader_contracts_ready"`
	SourceReaderWriteAllowed                                            bool   `json:"source_reader_write_allowed"`
	SourceReaderAdmissionAllowed                                        bool   `json:"source_reader_admission_allowed"`
	SourceReaderLiveAdmissionEnabled                                    bool   `json:"source_reader_live_admission_enabled"`
	SourceReaderMutatesState                                            bool   `json:"source_reader_mutates_state"`
	SourceReaderBodyTarget                                              string `json:"source_reader_body_target"`
	SourceReaderPassed                                                  bool   `json:"source_reader_passed"`
	SourceReaderReason                                                  string `json:"source_reader_reason"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID               string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady            bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID         string `json:"source_weighted_admission_resonance_graft_candidate_store_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreHash             string `json:"source_weighted_admission_resonance_graft_candidate_store_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash     string `json:"source_weighted_admission_resonance_graft_candidate_store_read_back_hash"`
	SourceStoreAction                                                   string `json:"source_store_action"`
	SourceStoreReceiptShape                                             string `json:"source_store_receipt_shape"`
	SourceStoreKind                                                     string `json:"source_store_kind"`
	SourceStoreMode                                                     string `json:"source_store_mode"`
	SourceStoreStage                                                    string `json:"source_store_stage"`
	SourceStoreAppendOnly                                               bool   `json:"source_store_append_only"`
	SourceStoreReadBack                                                 bool   `json:"source_store_read_back"`
	SourceStoreReceiptPersisted                                         bool   `json:"source_store_receipt_persisted"`
	SourceStoreReceiptVerified                                          bool   `json:"source_store_receipt_verified"`
	SourceStoreGraftAllowed                                             bool   `json:"source_store_graft_allowed"`
	SourceStoreRawDreamTextAllowed                                      bool   `json:"source_store_raw_dream_text_allowed"`
	SourceStoreJanusSurfaceAllowed                                      bool   `json:"source_store_janus_surface_allowed"`
	SourceStoreCoocLearningAllowed                                      bool   `json:"source_store_cooc_learning_allowed"`
	SourceStoreDeltaHarvestAllowed                                      bool   `json:"source_store_delta_harvest_allowed"`
	SourceStoreBodyMutationAllowed                                      bool   `json:"source_store_body_mutation_allowed"`
	SourceWeightedAdmissionResonanceGraftCandidateID                    string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady                 bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateCausalID              string `json:"source_weighted_admission_resonance_graft_candidate_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateHash                  string `json:"source_weighted_admission_resonance_graft_candidate_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateReadBackHash          string `json:"source_weighted_admission_resonance_graft_candidate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftGateID                         string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                      bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftGateCausalID                   string `json:"source_weighted_admission_resonance_graft_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftGateHash                       string `json:"source_weighted_admission_resonance_graft_gate_hash"`
	SourceWeightedAdmissionResonanceGraftGateReadBackHash               string `json:"source_weighted_admission_resonance_graft_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftPreflightID                    string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady                 bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                     string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady                  bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                       string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                    bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                          string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                       bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                         bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                               bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                    bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                                  bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                            bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                            bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                                   bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                    bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                 bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                        bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                             bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                              bool   `json:"source_authority_granted"`
	AuthorityGranted                                                    bool   `json:"authority_granted"`
	ContractsReady                                                      bool   `json:"contracts_ready"`
	WriteAllowed                                                        bool   `json:"write_allowed"`
	AdmissionAllowed                                                    bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                bool   `json:"live_admission_enabled"`
	MutatesState                                                        bool   `json:"mutates_state"`
	BodyTarget                                                          string `json:"body_target"`
	Passed                                                              bool   `json:"passed"`
	Reason                                                              string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProof(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_REPORT")
	}
	readerPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission proof output path missing")
	}
	reader, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportForAssert(readerPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportError(reader, root); err != nil {
		return err
	}
	proof := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport{
		Schema:       admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema,
		Status:       "shadow_graft_admission_proof_ready_dry_run",
		Target:       "resonance",
		TargetKind:   "weighted_internal_world_shadow_graft_admission_proof",
		TargetMode:   "receipt_only_closed_admission_proof_dry_run",
		Action:       "prove_weighted_resonance_shadow_graft_admission_dry_run",
		ReceiptShape: "weighted_resonance_shadow_graft_admission_proof_receipt",
		ProofKind:    "shadow_graft_admission_proof",
		ProofMode:    "closed_read_back_admission_proof",
		ProofStage:   "pre_live_graft_admission_proof",
		WeightedAdmissionResonanceGraftAdmissionProofReady:          true,
		WeightedAdmissionResonanceGraftCandidateStoreReaderConsumed: true,
		WeightedAdmissionResonanceGraftCandidateStoreReaderRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionProof:          true,
		StoreReaderVerified:    true,
		StoreVerified:          reader.StoreVerified,
		CandidateVerified:      reader.CandidateVerified,
		GateVerified:           reader.GateVerified,
		PreflightVerified:      reader.PreflightVerified,
		BoundaryVerified:       reader.BoundaryVerified,
		ObservationVerified:    reader.ObservationVerified,
		ReceiverVerified:       reader.ReceiverVerified,
		IntentVerified:         reader.IntentVerified,
		FinalGateVerified:      reader.FinalGateVerified,
		SealVerified:           reader.SealVerified,
		PermitVerified:         reader.PermitVerified,
		AuthorityVerified:      reader.AuthorityVerified,
		ReaderHashVerified:     true,
		ReaderReplayVerified:   true,
		ReaderReadBackVerified: true,
		AdmissionRequired:      true,
		ShadowOnly:             true,
		GraftAllowed:           false,
		DryRunOnly:             true,
		LiveReady:              true,
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
		SourceSchema:           reader.Schema,
		SourceStatus:           reader.Status,
		SourceTarget:           reader.Target,
		SourceReport:           readerPath,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:         reader.WeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:      reader.WeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID:   reader.CausalID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash:       reader.ReaderHash,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash: reader.ReplayHash,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack:   reader.ReadBackHash,
		SourceReaderAction:                                              reader.Action,
		SourceReaderReceiptShape:                                        reader.ReceiptShape,
		SourceReaderKind:                                                reader.ReaderKind,
		SourceReaderMode:                                                reader.ReaderMode,
		SourceReaderStage:                                               reader.ReaderStage,
		SourceReaderReadOnly:                                            reader.ReadOnly,
		SourceReaderReplayOnly:                                          reader.ReplayOnly,
		SourceReaderStoreVerified:                                       reader.StoreVerified,
		SourceReaderCandidateVerified:                                   reader.CandidateVerified,
		SourceReaderHashVerified:                                        true,
		SourceReaderReplayVerified:                                      true,
		SourceReaderReadBackVerified:                                    true,
		SourceReaderAdmissionRequired:                                   reader.AdmissionRequired,
		SourceReaderShadowOnly:                                          reader.ShadowOnly,
		SourceReaderGraftAllowed:                                        reader.GraftAllowed,
		SourceReaderDryRunOnly:                                          reader.DryRunOnly,
		SourceReaderLiveReady:                                           reader.LiveReady,
		SourceReaderRawDreamTextAllowed:                                 reader.RawDreamTextAllowed,
		SourceReaderRawDreamTextObserved:                                reader.RawDreamTextObserved,
		SourceReaderRawDreamTextForwarded:                               reader.RawDreamTextForwarded,
		SourceReaderJanusSurfaceAllowed:                                 reader.JanusSurfaceAllowed,
		SourceReaderCoocLearningAllowed:                                 reader.CoocLearningAllowed,
		SourceReaderDeltaHarvestAllowed:                                 reader.DeltaHarvestAllowed,
		SourceReaderBodyMutationAllowed:                                 reader.BodyMutationAllowed,
		SourceReaderRollbackRequired:                                    reader.RollbackRequired,
		SourceReaderAuthorityGranted:                                    reader.AuthorityGranted,
		SourceReaderContractsReady:                                      reader.ContractsReady,
		SourceReaderWriteAllowed:                                        reader.WriteAllowed,
		SourceReaderAdmissionAllowed:                                    reader.AdmissionAllowed,
		SourceReaderLiveAdmissionEnabled:                                reader.LiveAdmissionEnabled,
		SourceReaderMutatesState:                                        reader.MutatesState,
		SourceReaderBodyTarget:                                          reader.BodyTarget,
		SourceReaderPassed:                                              reader.Passed,
		SourceReaderReason:                                              reader.Reason,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:           reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:        reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID:     reader.SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreHash:         reader.SourceWeightedAdmissionResonanceGraftCandidateStoreHash,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash: reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		SourceStoreAction:                                               reader.SourceStoreAction,
		SourceStoreReceiptShape:                                         reader.SourceStoreReceiptShape,
		SourceStoreKind:                                                 reader.SourceStoreKind,
		SourceStoreMode:                                                 reader.SourceStoreMode,
		SourceStoreStage:                                                reader.SourceStoreStage,
		SourceStoreAppendOnly:                                           reader.SourceStoreAppendOnly,
		SourceStoreReadBack:                                             reader.SourceStoreReadBack,
		SourceStoreReceiptPersisted:                                     reader.SourceStoreReceiptPersisted,
		SourceStoreReceiptVerified:                                      reader.SourceStoreReceiptVerified,
		SourceStoreGraftAllowed:                                         reader.SourceStoreGraftAllowed,
		SourceStoreRawDreamTextAllowed:                                  reader.SourceStoreRawDreamTextAllowed,
		SourceStoreJanusSurfaceAllowed:                                  reader.SourceStoreJanusSurfaceAllowed,
		SourceStoreCoocLearningAllowed:                                  reader.SourceStoreCoocLearningAllowed,
		SourceStoreDeltaHarvestAllowed:                                  reader.SourceStoreDeltaHarvestAllowed,
		SourceStoreBodyMutationAllowed:                                  reader.SourceStoreBodyMutationAllowed,
		SourceWeightedAdmissionResonanceGraftCandidateID:                reader.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:             reader.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftCandidateCausalID:          reader.SourceWeightedAdmissionResonanceGraftCandidateCausalID,
		SourceWeightedAdmissionResonanceGraftCandidateHash:              reader.SourceWeightedAdmissionResonanceGraftCandidateHash,
		SourceWeightedAdmissionResonanceGraftCandidateReadBackHash:      reader.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash,
		SourceWeightedAdmissionResonanceGraftGateID:                     reader.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                  reader.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftGateCausalID:               reader.SourceWeightedAdmissionResonanceGraftGateCausalID,
		SourceWeightedAdmissionResonanceGraftGateHash:                   reader.SourceWeightedAdmissionResonanceGraftGateHash,
		SourceWeightedAdmissionResonanceGraftGateReadBackHash:           reader.SourceWeightedAdmissionResonanceGraftGateReadBackHash,
		SourceWeightedAdmissionResonanceGraftPreflightID:                reader.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:             reader.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                 reader.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:              reader.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                   reader.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                reader.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                      reader.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                   reader.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                     reader.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                           reader.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                reader.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                              reader.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                        reader.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                        reader.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                               reader.BodySmokeWeighted,
		NanoDirectRunner:                                                reader.NanoDirectRunner,
		NanoDirectFinalGate:                                             reader.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                    reader.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                         reader.BoundaryReportFullChain,
		SourceAuthorityGranted:                                          reader.SourceAuthorityGranted,
		AuthorityGranted:                                                false,
		ContractsReady:                                                  false,
		WriteAllowed:                                                    false,
		AdmissionAllowed:                                                false,
		LiveAdmissionEnabled:                                            false,
		MutatesState:                                                    false,
		BodyTarget:                                                      "none",
		Passed:                                                          true,
		Reason:                                                          "weighted resonance shadow graft admission proof sealed without body mutation",
	}
	proof.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofCausalID(proof)
	proof.ProofHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofHash(proof)
	proof.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReadBackHash(proof)
	proof.WeightedAdmissionResonanceGraftAdmissionProofID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofID(proof)
	if proof.CausalID == "" ||
		proof.ProofHash == "" ||
		proof.ReadBackHash == "" ||
		proof.WeightedAdmissionResonanceGraftAdmissionProofID == "" ||
		proof.ProofHash == proof.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof read-back proof failed")
	}
	raw, err := json.MarshalIndent(proof, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission proof marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission proof write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-proof] pass: resonance_graft_admission_proof_report=%s resonance_graft_candidate_store_reader_report=%s\n", outputPath, readerPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission proof schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema {
		return fmt.Errorf("weighted admission resonance graft admission proof schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema)
	}
	if report.Status != "shadow_graft_admission_proof_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof status mismatch: got %q want %q", report.Status, "shadow_graft_admission_proof_ready_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission proof target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission proof target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_proof")
	}
	if report.TargetMode != "receipt_only_closed_admission_proof_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_admission_proof_dry_run")
	}
	if report.Action != "prove_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof action mismatch: got %q want %q", report.Action, "prove_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission proof receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_proof_receipt")
	}
	if report.ProofKind != "shadow_graft_admission_proof" ||
		report.ProofMode != "closed_read_back_admission_proof" ||
		report.ProofStage != "pre_live_graft_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission proof shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_proof_ready", report.WeightedAdmissionResonanceGraftAdmissionProofReady},
		{"weighted_admission_resonance_graft_candidate_store_reader_consumed", report.WeightedAdmissionResonanceGraftCandidateStoreReaderConsumed},
		{"weighted_admission_resonance_graft_candidate_store_reader_required", report.WeightedAdmissionResonanceGraftCandidateStoreReaderRequired},
		{"next_step_blocked_without_resonance_graft_admission_proof", report.NextStepBlockedWithoutResonanceGraftAdmissionProof},
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
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady},
		{"source_reader_read_only", report.SourceReaderReadOnly},
		{"source_reader_replay_only", report.SourceReaderReplayOnly},
		{"source_reader_store_verified", report.SourceReaderStoreVerified},
		{"source_reader_candidate_verified", report.SourceReaderCandidateVerified},
		{"source_reader_hash_verified", report.SourceReaderHashVerified},
		{"source_reader_replay_verified", report.SourceReaderReplayVerified},
		{"source_reader_read_back_verified", report.SourceReaderReadBackVerified},
		{"source_reader_admission_required", report.SourceReaderAdmissionRequired},
		{"source_reader_shadow_only", report.SourceReaderShadowOnly},
		{"source_reader_dry_run_only", report.SourceReaderDryRunOnly},
		{"source_reader_live_ready", report.SourceReaderLiveReady},
		{"source_reader_rollback_required", report.SourceReaderRollbackRequired},
		{"source_reader_passed", report.SourceReaderPassed},
		{"source_weighted_admission_resonance_graft_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReady},
		{"source_store_append_only", report.SourceStoreAppendOnly},
		{"source_store_read_back", report.SourceStoreReadBack},
		{"source_store_receipt_persisted", report.SourceStoreReceiptPersisted},
		{"source_store_receipt_verified", report.SourceStoreReceiptVerified},
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
			return fmt.Errorf("weighted admission resonance graft admission proof %s not ready", required.name)
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
		{"source_reader_graft_allowed", report.SourceReaderGraftAllowed},
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
		{"source_store_graft_allowed", report.SourceStoreGraftAllowed},
		{"source_store_raw_dream_text_allowed", report.SourceStoreRawDreamTextAllowed},
		{"source_store_janus_surface_allowed", report.SourceStoreJanusSurfaceAllowed},
		{"source_store_cooc_learning_allowed", report.SourceStoreCoocLearningAllowed},
		{"source_store_delta_harvest_allowed", report.SourceStoreDeltaHarvestAllowed},
		{"source_store_body_mutation_allowed", report.SourceStoreBodyMutationAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission proof opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_proof_id", report.WeightedAdmissionResonanceGraftAdmissionProofID},
		{"causal_id", report.CausalID},
		{"proof_hash", report.ProofHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_causal_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_hash", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_replay_hash", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_read_back_hash", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack},
		{"source_weighted_admission_resonance_graft_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreID},
		{"source_weighted_admission_resonance_graft_candidate_store_causal_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID},
		{"source_weighted_admission_resonance_graft_candidate_store_hash", report.SourceWeightedAdmissionResonanceGraftCandidateStoreHash},
		{"source_weighted_admission_resonance_graft_candidate_store_read_back_hash", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash},
		{"source_weighted_admission_resonance_graft_candidate_id", report.SourceWeightedAdmissionResonanceGraftCandidateID},
		{"source_weighted_admission_resonance_graft_candidate_causal_id", report.SourceWeightedAdmissionResonanceGraftCandidateCausalID},
		{"source_weighted_admission_resonance_graft_candidate_hash", report.SourceWeightedAdmissionResonanceGraftCandidateHash},
		{"source_weighted_admission_resonance_graft_candidate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash},
		{"source_weighted_admission_resonance_graft_gate_id", report.SourceWeightedAdmissionResonanceGraftGateID},
		{"source_weighted_admission_resonance_graft_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftGateCausalID},
		{"source_weighted_admission_resonance_graft_gate_hash", report.SourceWeightedAdmissionResonanceGraftGateHash},
		{"source_weighted_admission_resonance_graft_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftGateReadBackHash},
		{"source_weighted_admission_resonance_graft_preflight_id", report.SourceWeightedAdmissionResonanceGraftPreflightID},
		{"source_weighted_admission_resonance_graft_boundary_id", report.SourceWeightedAdmissionResonanceGraftBoundaryID},
		{"source_weighted_admission_resonance_observation_id", report.SourceWeightedAdmissionResonanceObservationID},
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission proof %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema {
		return fmt.Errorf("weighted admission resonance graft admission proof source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema)
	}
	if report.SourceStatus != "shadow_graft_candidate_store_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_candidate_store_read_back_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceReaderAction != "read_weighted_resonance_shadow_graft_candidate_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_reader_action mismatch: got %q want %q", report.SourceReaderAction, "read_weighted_resonance_shadow_graft_candidate_store_dry_run")
	}
	if report.SourceReaderReceiptShape != "weighted_resonance_shadow_graft_candidate_store_reader_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_reader_receipt_shape mismatch: got %q want %q", report.SourceReaderReceiptShape, "weighted_resonance_shadow_graft_candidate_store_reader_receipt")
	}
	if report.SourceReaderKind != "shadow_graft_candidate_store_reader" ||
		report.SourceReaderMode != "read_only_replay" ||
		report.SourceReaderStage != "pre_live_graft_candidate_store_reader" {
		return fmt.Errorf("weighted admission resonance graft admission proof source reader shape mismatch")
	}
	if report.SourceStoreAction != "store_weighted_resonance_shadow_graft_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_store_action mismatch: got %q want %q", report.SourceStoreAction, "store_weighted_resonance_shadow_graft_candidate_dry_run")
	}
	if report.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_store_receipt_shape mismatch: got %q want %q", report.SourceStoreReceiptShape, "weighted_resonance_shadow_graft_candidate_store_receipt")
	}
	if report.SourceStoreKind != "shadow_graft_candidate_store" ||
		report.SourceStoreMode != "append_only_read_back_store" ||
		report.SourceStoreStage != "pre_live_graft_candidate_store" {
		return fmt.Errorf("weighted admission resonance graft admission proof source store shape mismatch")
	}
	if report.SourceReaderBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission proof source_reader_body_target mismatch: got %q want %q", report.SourceReaderBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission proof body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-proof-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission proof causal prefix mismatch")
	}
	if !strings.HasPrefix(report.ProofHash, "weighted-resonance-graft-admission-proof-") {
		return fmt.Errorf("weighted admission resonance graft admission proof hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-proof-read-") ||
		report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID, "weighted-resonance-graft-candidate-store-reader-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash, "weighted-resonance-graft-candidate-store-reader-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash, "weighted-resonance-graft-candidate-store-reader-replay-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack, "weighted-resonance-graft-candidate-store-reader-read-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source reader proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID, "weighted-resonance-graft-candidate-store-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreHash, "weighted-resonance-graft-candidate-store-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash, "weighted-resonance-graft-candidate-store-read-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source store proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateCausalID, "weighted-resonance-graft-candidate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateHash, "weighted-resonance-graft-candidate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash, "weighted-resonance-graft-candidate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateCausalID, "weighted-resonance-graft-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateHash, "weighted-resonance-graft-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateReadBackHash, "weighted-resonance-graft-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission proof source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof causal_id mismatch")
	}
	if report.ProofHash == "" || report.ProofHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof proof_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof read_back_hash mismatch")
	}
	if report.ProofHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionProofID == "" || report.WeightedAdmissionResonanceGraftAdmissionProofID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofID(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission proof sealed without body mutation" {
		return fmt.Errorf("weighted admission resonance graft admission proof reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofCausalID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport) string {
	h := hashJSON(struct {
		SourceReaderID       string `json:"source_reader_id"`
		SourceReaderReadBack string `json:"source_reader_read_back_hash"`
		SourceStoreID        string `json:"source_store_id"`
		SourceStoreReadBack  string `json:"source_store_read_back_hash"`
		SourceCandidateID    string `json:"source_candidate_id"`
		SourceGateID         string `json:"source_gate_id"`
		SourceObservationID  string `json:"source_observation_id"`
		Target               string `json:"target"`
		ProofKind            string `json:"proof_kind"`
		ProofStage           string `json:"proof_stage"`
	}{
		SourceReaderID:       proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceReaderReadBack: proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack,
		SourceStoreID:        proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceStoreReadBack:  proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		SourceCandidateID:    proof.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:         proof.SourceWeightedAdmissionResonanceGraftGateID,
		SourceObservationID:  proof.SourceWeightedAdmissionResonanceObservationID,
		Target:               proof.Target,
		ProofKind:            proof.ProofKind,
		ProofStage:           proof.ProofStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport) string {
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
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		DryRunOnly             bool   `json:"dry_run_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
	}{
		CausalID:               proof.CausalID,
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceReaderHash:       proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderHash,
		SourceReaderReplayHash: proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash,
		SourceReaderReadBack:   proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReadBack,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceStoreHash:        proof.SourceWeightedAdmissionResonanceGraftCandidateStoreHash,
		SourceStoreReadBack:    proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		ProofMode:              proof.ProofMode,
		ReceiptShape:           proof.ReceiptShape,
		StoreReaderVerified:    proof.StoreReaderVerified,
		StoreVerified:          proof.StoreVerified,
		CandidateVerified:      proof.CandidateVerified,
		ReaderHashVerified:     proof.ReaderHashVerified,
		ReaderReplayVerified:   proof.ReaderReplayVerified,
		ReaderReadBackVerified: proof.ReaderReadBackVerified,
		ReadOnly:               proof.ReadOnly,
		ReplayOnly:             proof.ReplayOnly,
		AdmissionRequired:      proof.AdmissionRequired,
		ShadowOnly:             proof.ShadowOnly,
		DryRunOnly:             proof.DryRunOnly,
		GraftAllowed:           proof.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReadBackHash(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport) string {
	h := hashJSON(struct {
		ProofHash        string `json:"proof_hash"`
		SourceReaderID   string `json:"source_reader_id"`
		SourceStoreID    string `json:"source_store_id"`
		SourceCandidate  string `json:"source_candidate_id"`
		ProofKind        string `json:"proof_kind"`
		ProofReady       bool   `json:"proof_ready"`
		BodyMutation     bool   `json:"body_mutation"`
		LiveAdmission    bool   `json:"live_admission"`
		WriteAllowed     bool   `json:"write_allowed"`
		AdmissionAllowed bool   `json:"admission_allowed"`
	}{
		ProofHash:        proof.ProofHash,
		SourceReaderID:   proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:    proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidate:  proof.SourceWeightedAdmissionResonanceGraftCandidateID,
		ProofKind:        proof.ProofKind,
		ProofReady:       proof.WeightedAdmissionResonanceGraftAdmissionProofReady,
		BodyMutation:     proof.BodyMutationAllowed,
		LiveAdmission:    proof.LiveAdmissionEnabled,
		WriteAllowed:     proof.WriteAllowed,
		AdmissionAllowed: proof.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofID(proof admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport) string {
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
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
		DryRunOnly             bool   `json:"dry_run_only"`
		RawDreamTextAllowed    bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed    bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed    bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed    bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		RollbackRequired       bool   `json:"rollback_required"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		LiveReady              bool   `json:"live_ready"`
		ContractsReady         bool   `json:"contracts_ready"`
		BodyTarget             string `json:"body_target"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_proof"`
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
		SourcePermitReady      bool   `json:"source_permit_ready"`
		SourceAuthorityUsed    bool   `json:"source_authority_consumed"`
		SourceAuthorityNeeded  bool   `json:"source_authority_required"`
	}{
		Schema:                 proof.Schema,
		Status:                 proof.Status,
		Action:                 proof.Action,
		SourceReport:           proof.SourceReport,
		SourceReaderID:         proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:          proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:      proof.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:           proof.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:      proof.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:       proof.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:    proof.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:       proof.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:               proof.CausalID,
		ProofHash:              proof.ProofHash,
		ReadBackHash:           proof.ReadBackHash,
		Ready:                  proof.WeightedAdmissionResonanceGraftAdmissionProofReady,
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
		AdmissionRequired:      proof.AdmissionRequired,
		ShadowOnly:             proof.ShadowOnly,
		GraftAllowed:           proof.GraftAllowed,
		DryRunOnly:             proof.DryRunOnly,
		RawDreamTextAllowed:    proof.RawDreamTextAllowed,
		JanusSurfaceAllowed:    proof.JanusSurfaceAllowed,
		CoocLearningAllowed:    proof.CoocLearningAllowed,
		DeltaHarvestAllowed:    proof.DeltaHarvestAllowed,
		BodyMutationAllowed:    proof.BodyMutationAllowed,
		RollbackRequired:       proof.RollbackRequired,
		ReadOnly:               proof.ReadOnly,
		ReplayOnly:             proof.ReplayOnly,
		LiveReady:              proof.LiveReady,
		ContractsReady:         proof.ContractsReady,
		BodyTarget:             proof.BodyTarget,
		WriteAllowed:           proof.WriteAllowed,
		AdmissionAllowed:       proof.AdmissionAllowed,
		LiveAdmissionEnabled:   proof.LiveAdmissionEnabled,
		MutatesState:           proof.MutatesState,
		NextStepBlockedWithout: proof.NextStepBlockedWithoutResonanceGraftAdmissionProof,
		SourceReaderReady:      proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:       proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:   proof.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:        proof.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:   proof.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:    proof.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady: proof.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:    proof.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:      proof.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:   proof.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:        proof.SourceWeightedAdmissionSealReady,
		SourcePermitReady:      proof.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:    proof.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:  proof.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission proof not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof decode failed: %w", err)
	}
	return report, root, nil
}
