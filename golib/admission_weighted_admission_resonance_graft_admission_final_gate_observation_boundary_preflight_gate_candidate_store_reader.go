package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport struct {
	Schema                                                                                                     string `json:"schema"`
	Status                                                                                                     string `json:"status"`
	Target                                                                                                     string `json:"target"`
	TargetKind                                                                                                 string `json:"target_kind"`
	TargetMode                                                                                                 string `json:"target_mode"`
	Action                                                                                                     string `json:"action"`
	LedgerState                                                                                                string `json:"ledger_state"`
	LedgerAction                                                                                               string `json:"ledger_action"`
	LedgerContract                                                                                             string `json:"ledger_contract"`
	LedgerEntrypoint                                                                                           string `json:"ledger_entrypoint"`
	LedgerReceiptShape                                                                                         string `json:"ledger_receipt_shape"`
	LedgerWriteScope                                                                                           string `json:"ledger_write_scope"`
	LedgerReady                                                                                                bool   `json:"ledger_ready"`
	LedgerAppendAllowed                                                                                        bool   `json:"ledger_append_allowed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreConsumed    bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreRequired    bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id"`
	ReceiptShape                                                                                               string `json:"receipt_shape"`
	ReaderKind                                                                                                 string `json:"reader_kind"`
	ReaderMode                                                                                                 string `json:"reader_mode"`
	ReaderStage                                                                                                string `json:"reader_stage"`
	CausalID                                                                                                   string `json:"causal_id"`
	ReaderHash                                                                                                 string `json:"reader_hash"`
	ReplayHash                                                                                                 string `json:"replay_hash"`
	ReadBackHash                                                                                               string `json:"read_back_hash"`
	StoreVerified                                                                                              bool   `json:"store_verified"`
	CandidateVerified                                                                                          bool   `json:"candidate_verified"`
	GateVerified                                                                                               bool   `json:"gate_verified"`
	PreflightVerified                                                                                          bool   `json:"preflight_verified"`
	BoundaryVerified                                                                                           bool   `json:"boundary_verified"`
	ObservationVerified                                                                                        bool   `json:"observation_verified"`
	FinalGateVerified                                                                                          bool   `json:"final_gate_verified"`
	SealVerified                                                                                               bool   `json:"seal_verified"`
	PermitVerified                                                                                             bool   `json:"permit_verified"`
	AuthorityVerified                                                                                          bool   `json:"authority_verified"`
	StoreHashVerified                                                                                          bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                                                                      bool   `json:"store_read_back_verified"`
	AdmissionRequired                                                                                          bool   `json:"admission_required"`
	ShadowOnly                                                                                                 bool   `json:"shadow_only"`
	DryRunOnly                                                                                                 bool   `json:"dry_run_only"`
	LiveReady                                                                                                  bool   `json:"live_ready"`
	RollbackRequired                                                                                           bool   `json:"rollback_required"`
	ReadOnly                                                                                                   bool   `json:"read_only"`
	ReplayOnly                                                                                                 bool   `json:"replay_only"`
	RawDreamTextAllowed                                                                                        bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                                                       bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                                                      bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                                                        bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                                                        bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                                                        bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                                                        bool   `json:"body_mutation_allowed"`
	AuthorityGranted                                                                                           bool   `json:"authority_granted"`
	ContractsReady                                                                                             bool   `json:"contracts_ready"`
	WriteAllowed                                                                                               bool   `json:"write_allowed"`
	AdmissionAllowed                                                                                           bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                                                       bool   `json:"live_admission_enabled"`
	MutatesState                                                                                               bool   `json:"mutates_state"`
	BodyTarget                                                                                                 string `json:"body_target"`
	Passed                                                                                                     bool   `json:"passed"`
	Reason                                                                                                     string `json:"reason"`

	SourceSchema                                                                                                string `json:"source_schema"`
	SourceStatus                                                                                                string `json:"source_status"`
	SourceTarget                                                                                                string `json:"source_target"`
	SourceReport                                                                                                string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash                          string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_hash"`
	SourceStoreReceiptShape                                                                                     string `json:"source_store_receipt_shape"`
	SourceStoreKind                                                                                             string `json:"source_store_kind"`
	SourceStoreMode                                                                                             string `json:"source_store_mode"`
	SourceStoreStage                                                                                            string `json:"source_store_stage"`
	SourceStoreAppendOnly                                                                                       bool   `json:"source_store_append_only"`
	SourceStoreReadBack                                                                                         bool   `json:"source_store_read_back"`
	SourceStoreReceiptPersisted                                                                                 bool   `json:"source_store_receipt_persisted"`
	SourceStoreReceiptVerified                                                                                  bool   `json:"source_store_receipt_verified"`
	SourceStoreAdmissionRequired                                                                                bool   `json:"source_store_admission_required"`
	SourceStoreShadowOnly                                                                                       bool   `json:"source_store_shadow_only"`
	SourceStoreDryRunOnly                                                                                       bool   `json:"source_store_dry_run_only"`
	SourceStoreLiveReady                                                                                        bool   `json:"source_store_live_ready"`
	SourceStoreRollbackRequired                                                                                 bool   `json:"source_store_rollback_required"`
	SourceStoreLedgerState                                                                                      string `json:"source_store_ledger_state"`
	SourceStoreLedgerAction                                                                                     string `json:"source_store_ledger_action"`
	SourceStoreLedgerContract                                                                                   string `json:"source_store_ledger_contract"`
	SourceStoreLedgerEntrypoint                                                                                 string `json:"source_store_ledger_entrypoint"`
	SourceStoreLedgerReceiptShape                                                                               string `json:"source_store_ledger_receipt_shape"`
	SourceStoreLedgerWriteScope                                                                                 string `json:"source_store_ledger_write_scope"`
	SourceStoreLedgerReady                                                                                      bool   `json:"source_store_ledger_ready"`
	SourceStoreLedgerAppendAllowed                                                                              bool   `json:"source_store_ledger_append_allowed"`
	SourceStoreRawDreamTextAllowed                                                                              bool   `json:"source_store_raw_dream_text_allowed"`
	SourceStoreRawDreamTextObserved                                                                             bool   `json:"source_store_raw_dream_text_observed"`
	SourceStoreRawDreamTextForwarded                                                                            bool   `json:"source_store_raw_dream_text_forwarded"`
	SourceStoreJanusSurfaceAllowed                                                                              bool   `json:"source_store_janus_surface_allowed"`
	SourceStoreCoocLearningAllowed                                                                              bool   `json:"source_store_cooc_learning_allowed"`
	SourceStoreDeltaHarvestAllowed                                                                              bool   `json:"source_store_delta_harvest_allowed"`
	SourceStoreBodyMutationAllowed                                                                              bool   `json:"source_store_body_mutation_allowed"`
	SourceStoreAuthorityGranted                                                                                 bool   `json:"source_store_authority_granted"`
	SourceStoreContractsReady                                                                                   bool   `json:"source_store_contracts_ready"`
	SourceStoreWriteAllowed                                                                                     bool   `json:"source_store_write_allowed"`
	SourceStoreAdmissionAllowed                                                                                 bool   `json:"source_store_admission_allowed"`
	SourceStoreLiveAdmissionEnabled                                                                             bool   `json:"source_store_live_admission_enabled"`
	SourceStoreMutatesState                                                                                     bool   `json:"source_store_mutates_state"`
	SourceStoreBodyTarget                                                                                       string `json:"source_store_body_target"`

	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID       string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady    bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash                                    string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash                            string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason                                  string `json:"source_admission_final_gate_observation_boundary_preflight_gate_candidate_reason"`
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
	SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified                       bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified                        bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified                     bool   `json:"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT")
	}
	storePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader output path missing")
	}
	store, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReportForAssert(storePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReportError(store, root); err != nil {
		return err
	}
	reader := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport{
		Schema:              admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema,
		Status:              "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run",
		Target:              "live_route_admission_next_step",
		TargetKind:          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader",
		TargetMode:          "read_only_replay_dry_run",
		Action:              "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run",
		LedgerState:         "blocked",
		LedgerAction:        "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ledger_append",
		LedgerContract:      "none",
		LedgerEntrypoint:    "none",
		LedgerReceiptShape:  "none",
		LedgerWriteScope:    "none",
		LedgerReady:         false,
		LedgerAppendAllowed: false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady: true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreConsumed:    true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreRequired:    true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader: true,
		ReceiptShape:          "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt",
		ReaderKind:            "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader",
		ReaderMode:            "read_only_replay",
		ReaderStage:           "post_preflight_gate_candidate_store_pre_live_admission_reader",
		StoreVerified:         true,
		CandidateVerified:     store.CandidateVerified,
		GateVerified:          store.GateVerified,
		PreflightVerified:     store.PreflightVerified,
		BoundaryVerified:      store.BoundaryVerified,
		ObservationVerified:   store.ObservationVerified,
		FinalGateVerified:     store.FinalGateVerified,
		SealVerified:          store.SealVerified,
		PermitVerified:        store.PermitVerified,
		AuthorityVerified:     store.AuthorityVerified,
		StoreHashVerified:     true,
		StoreReadBackVerified: true,
		AdmissionRequired:     true,
		ShadowOnly:            true,
		DryRunOnly:            true,
		LiveReady:             store.LiveReady,
		RollbackRequired:      true,
		ReadOnly:              true,
		ReplayOnly:            true,
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
		Reason:                "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store read back without ledger append or body mutation",

		SourceSchema: store.Schema,
		SourceStatus: store.Status,
		SourceTarget: store.Target,
		SourceReport: storePath,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID:     store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady:  store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal: store.CausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash:                                  store.StoreHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash:                          store.ReadBackHash,
		SourceStoreReceiptShape:          store.ReceiptShape,
		SourceStoreKind:                  store.StoreKind,
		SourceStoreMode:                  store.StoreMode,
		SourceStoreStage:                 store.StoreStage,
		SourceStoreAppendOnly:            store.AppendOnly,
		SourceStoreReadBack:              store.ReadBack,
		SourceStoreReceiptPersisted:      store.ReceiptPersisted,
		SourceStoreReceiptVerified:       store.ReceiptVerified,
		SourceStoreAdmissionRequired:     store.AdmissionRequired,
		SourceStoreShadowOnly:            store.ShadowOnly,
		SourceStoreDryRunOnly:            store.DryRunOnly,
		SourceStoreLiveReady:             store.LiveReady,
		SourceStoreRollbackRequired:      store.RollbackRequired,
		SourceStoreLedgerState:           store.LedgerState,
		SourceStoreLedgerAction:          store.LedgerAction,
		SourceStoreLedgerContract:        store.LedgerContract,
		SourceStoreLedgerEntrypoint:      store.LedgerEntrypoint,
		SourceStoreLedgerReceiptShape:    store.LedgerReceiptShape,
		SourceStoreLedgerWriteScope:      store.LedgerWriteScope,
		SourceStoreLedgerReady:           store.LedgerReady,
		SourceStoreLedgerAppendAllowed:   store.LedgerAppendAllowed,
		SourceStoreRawDreamTextAllowed:   false,
		SourceStoreRawDreamTextObserved:  false,
		SourceStoreRawDreamTextForwarded: false,
		SourceStoreJanusSurfaceAllowed:   false,
		SourceStoreCoocLearningAllowed:   false,
		SourceStoreDeltaHarvestAllowed:   false,
		SourceStoreBodyMutationAllowed:   false,
		SourceStoreAuthorityGranted:      store.AuthorityGranted,
		SourceStoreContractsReady:        store.ContractsReady,
		SourceStoreWriteAllowed:          store.WriteAllowed,
		SourceStoreAdmissionAllowed:      store.AdmissionAllowed,
		SourceStoreLiveAdmissionEnabled:  store.LiveAdmissionEnabled,
		SourceStoreMutatesState:          store.MutatesState,
		SourceStoreBodyTarget:            store.BodyTarget,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID:       store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady:    store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID: store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash:                                    store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash:                            store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason:                                  store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason,
		SourceCandidateReceiptShape:          store.SourceCandidateReceiptShape,
		SourceCandidateState:                 store.SourceCandidateState,
		SourceCandidateKind:                  store.SourceCandidateKind,
		SourceCandidateMode:                  store.SourceCandidateMode,
		SourceCandidateStage:                 store.SourceCandidateStage,
		SourceCandidateDryRunOnly:            store.SourceCandidateDryRunOnly,
		SourceCandidateGateVerified:          store.SourceCandidateGateVerified,
		SourceCandidatePreflightVerified:     store.SourceCandidatePreflightVerified,
		SourceCandidateBoundaryVerified:      store.SourceCandidateBoundaryVerified,
		SourceCandidateObservationVerified:   store.SourceCandidateObservationVerified,
		SourceCandidateReadBackVerified:      store.SourceCandidateReadBackVerified,
		SourceCandidateOpened:                store.SourceCandidateOpened,
		SourceCandidateRawDreamTextObserved:  store.SourceCandidateRawDreamTextObserved,
		SourceCandidateRawDreamTextForwarded: store.SourceCandidateRawDreamTextForwarded,
		SourceCandidateRawDreamTextAllowed:   store.SourceCandidateRawDreamTextAllowed,
		SourceCandidateBodyMutationAllowed:   store.SourceCandidateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID:       store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:    store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID: store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateHash:                                    store.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash:                            store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReady:                                   store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:                              store.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified:                       store.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified:                        store.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified:                     store.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified:                        store.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved:                             store.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:                            store.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:                              store.SourceFinalGateObservationBoundaryPreflightGateRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed:                              store.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed,

		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:    store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady: store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID:             store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:          store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                     store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                  store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                        store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:                     store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:                       store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:                             store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:                                  store.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:                             store.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:                                store.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:                             store.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWriterInventoryVerified:                                                            store.SourceWriterInventoryVerified,
		SourceWriterPreflightVerified:                                                            store.SourceWriterPreflightVerified,
		SourceAdmissionRequired:                                                                  store.SourceAdmissionRequired,
		SourceShadowOnly:                                                                         store.SourceShadowOnly,
		SourceDryRunOnly:                                                                         store.SourceDryRunOnly,
		SourceRequiresWriter:                                                                     store.SourceRequiresWriter,
		SourceRollbackRequired:                                                                   store.SourceRollbackRequired,
		SourceRequiresRollback:                                                                   store.SourceRequiresRollback,
		SourceReadOnly:                                                                           store.SourceReadOnly,
		SourceReplayOnly:                                                                         store.SourceReplayOnly,
	}
	reader.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID(reader)
	reader.ReaderHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash(reader)
	reader.ReplayHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash(reader)
	reader.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash(reader)
	reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID(reader)
	if reader.CausalID == "" ||
		reader.ReaderHash == "" ||
		reader.ReplayHash == "" ||
		reader.ReadBackHash == "" ||
		reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID == "" ||
		reader.ReaderHash == reader.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader read-back proof failed")
	}
	raw, err := json.MarshalIndent(reader, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_report=%s\n", outputPath, storePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run")
	}
	if report.Target != "live_route_admission_next_step" ||
		report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		report.TargetMode != "read_only_replay_dry_run" ||
		report.Action != "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader route shape mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ledger_append" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader ledger guard mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt" ||
		report.ReaderKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		report.ReaderMode != "read_only_replay" ||
		report.ReaderStage != "post_preflight_gate_candidate_store_pre_live_admission_reader" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader},
		{"store_verified", report.StoreVerified},
		{"candidate_verified", report.CandidateVerified},
		{"gate_verified", report.GateVerified},
		{"preflight_verified", report.PreflightVerified},
		{"boundary_verified", report.BoundaryVerified},
		{"observation_verified", report.ObservationVerified},
		{"final_gate_verified", report.FinalGateVerified},
		{"seal_verified", report.SealVerified},
		{"permit_verified", report.PermitVerified},
		{"authority_verified", report.AuthorityVerified},
		{"store_hash_verified", report.StoreHashVerified},
		{"store_read_back_verified", report.StoreReadBackVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady},
		{"source_store_append_only", report.SourceStoreAppendOnly},
		{"source_store_read_back", report.SourceStoreReadBack},
		{"source_store_receipt_persisted", report.SourceStoreReceiptPersisted},
		{"source_store_receipt_verified", report.SourceStoreReceiptVerified},
		{"source_store_admission_required", report.SourceStoreAdmissionRequired},
		{"source_store_shadow_only", report.SourceStoreShadowOnly},
		{"source_store_dry_run_only", report.SourceStoreDryRunOnly},
		{"source_store_live_ready", report.SourceStoreLiveReady},
		{"source_store_rollback_required", report.SourceStoreRollbackRequired},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady},
		{"source_candidate_dry_run_only", report.SourceCandidateDryRunOnly},
		{"source_candidate_gate_verified", report.SourceCandidateGateVerified},
		{"source_candidate_preflight_verified", report.SourceCandidatePreflightVerified},
		{"source_candidate_boundary_verified", report.SourceCandidateBoundaryVerified},
		{"source_candidate_observation_verified", report.SourceCandidateObservationVerified},
		{"source_candidate_read_back_verified", report.SourceCandidateReadBackVerified},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"source_admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_gate_preflight_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_boundary_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified},
		{"source_admission_final_gate_observation_boundary_preflight_gate_observation_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID},
		{"causal_id", report.CausalID},
		{"reader_hash", report.ReaderHash},
		{"replay_hash", report.ReplayHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run" ||
		report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source route mismatch")
	}
	if report.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_receipt" ||
		report.SourceStoreKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store" ||
		report.SourceStoreMode != "append_only_read_back_store" ||
		report.SourceStoreStage != "post_preflight_gate_candidate_pre_live_admission_store" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source store shape mismatch")
	}
	if report.SourceStoreLedgerState != "blocked" ||
		report.SourceStoreLedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ledger_append" ||
		report.SourceStoreLedgerContract != "none" ||
		report.SourceStoreLedgerEntrypoint != "none" ||
		report.SourceStoreLedgerReceiptShape != "none" ||
		report.SourceStoreLedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source store ledger guard mismatch")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source candidate reason mismatch: got %q", report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReason)
	}
	if report.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_receipt" ||
		report.SourceCandidateState != "blocked" ||
		report.SourceCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		report.SourceCandidateMode != "no_mutation_preflight_gate_candidate" ||
		report.SourceCandidateStage != "post_preflight_gate_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source candidate shape mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-") ||
		!strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-causal-") ||
		!strings.HasPrefix(report.ReaderHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-") ||
		!strings.HasPrefix(report.ReplayHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-replay-") ||
		!strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-read-") ||
		report.ReaderHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader proof prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source store proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID, "weighted-resonance-graft-admission-final-gate-observation-boundary-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID, "weighted-resonance-graft-admission-final-gate-observation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-receiver-id-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source chain prefix mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.SourceStoreBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader source_store_body_target mismatch: got %q want %q", report.SourceStoreBodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader causal_id mismatch")
	}
	if report.ReaderHash == "" || report.ReaderHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader reader_hash mismatch")
	}
	if report.ReplayHash == "" || report.ReplayHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader replay_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader read_back_hash mismatch")
	}
	if report.ReaderHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID == "" ||
		report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store read back without ledger append or body mutation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID(reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		SourceStoreID       string `json:"source_store_id"`
		SourceStoreReadBack string `json:"source_store_read_back_hash"`
		SourceCandidateID   string `json:"source_candidate_id"`
		SourceGateID        string `json:"source_gate_id"`
		SourcePreflightID   string `json:"source_preflight_id"`
		SourceBoundaryID    string `json:"source_boundary_id"`
		SourceObservationID string `json:"source_observation_id"`
		Target              string `json:"target"`
		ReaderKind          string `json:"reader_kind"`
		ReaderStage         string `json:"reader_stage"`
	}{
		SourceStoreID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceStoreReadBack: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		SourceCandidateID:   reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceGateID:        reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourcePreflightID:   reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceBoundaryID:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceObservationID: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		Target:              reader.Target,
		ReaderKind:          reader.ReaderKind,
		ReaderStage:         reader.ReaderStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		CausalID       string `json:"causal_id"`
		SourceStoreID  string `json:"source_store_id"`
		SourceHash     string `json:"source_hash"`
		SourceReadBack string `json:"source_read_back_hash"`
		ReaderMode     string `json:"reader_mode"`
		ReceiptShape   string `json:"receipt_shape"`
		ReadOnly       bool   `json:"read_only"`
		ReplayOnly     bool   `json:"replay_only"`
		StoreVerified  bool   `json:"store_verified"`
		SourceAppend   bool   `json:"source_append_only"`
		SourceRead     bool   `json:"source_read_back"`
		SourceVerified bool   `json:"source_receipt_verified"`
		LedgerAppend   bool   `json:"ledger_append_allowed"`
		BodyMutation   bool   `json:"body_mutation"`
		LiveAdmission  bool   `json:"live_admission"`
	}{
		CausalID:       reader.CausalID,
		SourceStoreID:  reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceHash:     reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceReadBack: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash,
		ReaderMode:     reader.ReaderMode,
		ReceiptShape:   reader.ReceiptShape,
		ReadOnly:       reader.ReadOnly,
		ReplayOnly:     reader.ReplayOnly,
		StoreVerified:  reader.StoreVerified,
		SourceAppend:   reader.SourceStoreAppendOnly,
		SourceRead:     reader.SourceStoreReadBack,
		SourceVerified: reader.SourceStoreReceiptVerified,
		LedgerAppend:   reader.LedgerAppendAllowed,
		BodyMutation:   reader.BodyMutationAllowed,
		LiveAdmission:  reader.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		ReaderHash      string `json:"reader_hash"`
		SourceStoreID   string `json:"source_store_id"`
		SourceStoreHash string `json:"source_store_hash"`
		SourceCandidate string `json:"source_candidate_id"`
		ReadOnly        bool   `json:"read_only"`
		ReplayOnly      bool   `json:"replay_only"`
		StoreVerified   bool   `json:"store_verified"`
	}{
		ReaderHash:      reader.ReaderHash,
		SourceStoreID:   reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceStoreHash: reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash,
		SourceCandidate: reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		ReadOnly:        reader.ReadOnly,
		ReplayOnly:      reader.ReplayOnly,
		StoreVerified:   reader.StoreVerified,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-replay-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		ReaderHash     string `json:"reader_hash"`
		ReplayHash     string `json:"replay_hash"`
		SourceStore    string `json:"source_store_id"`
		ReaderKind     string `json:"reader_kind"`
		ReaderReady    bool   `json:"reader_ready"`
		ReadOnly       bool   `json:"read_only"`
		BodyMutation   bool   `json:"body_mutation"`
		AdmissionOpen  bool   `json:"admission_open"`
		LedgerAppend   bool   `json:"ledger_append_allowed"`
		SourceVerified bool   `json:"source_receipt_verified"`
	}{
		ReaderHash:     reader.ReaderHash,
		ReplayHash:     reader.ReplayHash,
		SourceStore:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		ReaderKind:     reader.ReaderKind,
		ReaderReady:    reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		ReadOnly:       reader.ReadOnly,
		BodyMutation:   reader.BodyMutationAllowed,
		AdmissionOpen:  reader.LiveAdmissionEnabled,
		LedgerAppend:   reader.LedgerAppendAllowed,
		SourceVerified: reader.SourceStoreReceiptVerified,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID(reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceStoreID          string `json:"source_store_id"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		SourcePreflightID      string `json:"source_preflight_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		CausalID               string `json:"causal_id"`
		ReaderHash             string `json:"reader_hash"`
		ReplayHash             string `json:"replay_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReaderKind             string `json:"reader_kind"`
		ReaderMode             string `json:"reader_mode"`
		ReaderStage            string `json:"reader_stage"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		StoreVerified          bool   `json:"store_verified"`
		StoreHashVerified      bool   `json:"store_hash_verified"`
		StoreReadBackVerified  bool   `json:"store_read_back_verified"`
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
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader"`
		SourceStoreReady       bool   `json:"source_store_ready"`
		SourceCandidateOpened  bool   `json:"source_candidate_opened"`
	}{
		Schema:                 reader.Schema,
		Status:                 reader.Status,
		Action:                 reader.Action,
		SourceReport:           reader.SourceReport,
		SourceStoreID:          reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID,
		SourceCandidateID:      reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID,
		SourceGateID:           reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID,
		SourcePreflightID:      reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceBoundaryID:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID,
		SourceObservationID:    reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		CausalID:               reader.CausalID,
		ReaderHash:             reader.ReaderHash,
		ReplayHash:             reader.ReplayHash,
		ReadBackHash:           reader.ReadBackHash,
		Ready:                  reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady,
		ReaderKind:             reader.ReaderKind,
		ReaderMode:             reader.ReaderMode,
		ReaderStage:            reader.ReaderStage,
		ReadOnly:               reader.ReadOnly,
		ReplayOnly:             reader.ReplayOnly,
		StoreVerified:          reader.StoreVerified,
		StoreHashVerified:      reader.StoreHashVerified,
		StoreReadBackVerified:  reader.StoreReadBackVerified,
		AdmissionRequired:      reader.AdmissionRequired,
		ShadowOnly:             reader.ShadowOnly,
		DryRunOnly:             reader.DryRunOnly,
		LiveReady:              reader.LiveReady,
		LedgerAppendAllowed:    reader.LedgerAppendAllowed,
		BodyTarget:             reader.BodyTarget,
		WriteAllowed:           reader.WriteAllowed,
		AdmissionAllowed:       reader.AdmissionAllowed,
		LiveAdmissionEnabled:   reader.LiveAdmissionEnabled,
		MutatesState:           reader.MutatesState,
		NextStepBlockedWithout: reader.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader,
		SourceStoreReady:       reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReady,
		SourceCandidateOpened:  reader.SourceCandidateOpened,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader decode failed: %w", err)
	}
	return report, root, nil
}
