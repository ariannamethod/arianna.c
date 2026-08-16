package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema = "arianna.live_route_weighted_admission_resonance_graft_candidate_store_reader.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport struct {
	Schema                                                          string `json:"schema"`
	Status                                                          string `json:"status"`
	Target                                                          string `json:"target"`
	TargetKind                                                      string `json:"target_kind"`
	TargetMode                                                      string `json:"target_mode"`
	Action                                                          string `json:"action"`
	WeightedAdmissionResonanceGraftCandidateStoreReaderReady        bool   `json:"weighted_admission_resonance_graft_candidate_store_reader_ready"`
	WeightedAdmissionResonanceGraftCandidateStoreConsumed           bool   `json:"weighted_admission_resonance_graft_candidate_store_consumed"`
	WeightedAdmissionResonanceGraftCandidateStoreRequired           bool   `json:"weighted_admission_resonance_graft_candidate_store_required"`
	NextStepBlockedWithoutResonanceGraftCandidateStoreReader        bool   `json:"next_step_blocked_without_resonance_graft_candidate_store_reader"`
	WeightedAdmissionResonanceGraftCandidateStoreReaderID           string `json:"weighted_admission_resonance_graft_candidate_store_reader_id"`
	ReceiptShape                                                    string `json:"receipt_shape"`
	ReaderKind                                                      string `json:"reader_kind"`
	ReaderMode                                                      string `json:"reader_mode"`
	ReaderStage                                                     string `json:"reader_stage"`
	CausalID                                                        string `json:"causal_id"`
	ReaderHash                                                      string `json:"reader_hash"`
	ReplayHash                                                      string `json:"replay_hash"`
	ReadBackHash                                                    string `json:"read_back_hash"`
	StoreVerified                                                   bool   `json:"store_verified"`
	CandidateVerified                                               bool   `json:"candidate_verified"`
	GateVerified                                                    bool   `json:"gate_verified"`
	PreflightVerified                                               bool   `json:"preflight_verified"`
	BoundaryVerified                                                bool   `json:"boundary_verified"`
	ObservationVerified                                             bool   `json:"observation_verified"`
	ReceiverVerified                                                bool   `json:"receiver_verified"`
	IntentVerified                                                  bool   `json:"intent_verified"`
	FinalGateVerified                                               bool   `json:"final_gate_verified"`
	SealVerified                                                    bool   `json:"seal_verified"`
	PermitVerified                                                  bool   `json:"permit_verified"`
	AuthorityVerified                                               bool   `json:"authority_verified"`
	StoreHashVerified                                               bool   `json:"store_hash_verified"`
	StoreReadBackVerified                                           bool   `json:"store_read_back_verified"`
	AdmissionRequired                                               bool   `json:"admission_required"`
	ShadowOnly                                                      bool   `json:"shadow_only"`
	GraftAllowed                                                    bool   `json:"graft_allowed"`
	DryRunOnly                                                      bool   `json:"dry_run_only"`
	LiveReady                                                       bool   `json:"live_ready"`
	RawDreamTextAllowed                                             bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                            bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                           bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                             bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                             bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                             bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                             bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                bool   `json:"rollback_required"`
	ReadOnly                                                        bool   `json:"read_only"`
	ReplayOnly                                                      bool   `json:"replay_only"`
	SourceAppendOnly                                                bool   `json:"source_append_only"`
	SourceReadBack                                                  bool   `json:"source_read_back"`
	SourceReceiptPersisted                                          bool   `json:"source_receipt_persisted"`
	SourceReceiptVerified                                           bool   `json:"source_receipt_verified"`
	SourceSchema                                                    string `json:"source_schema"`
	SourceStatus                                                    string `json:"source_status"`
	SourceTarget                                                    string `json:"source_target"`
	SourceReport                                                    string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID           string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady        bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID     string `json:"source_weighted_admission_resonance_graft_candidate_store_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreHash         string `json:"source_weighted_admission_resonance_graft_candidate_store_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash string `json:"source_weighted_admission_resonance_graft_candidate_store_read_back_hash"`
	SourceStoreAction                                               string `json:"source_store_action"`
	SourceStoreReceiptShape                                         string `json:"source_store_receipt_shape"`
	SourceStoreKind                                                 string `json:"source_store_kind"`
	SourceStoreMode                                                 string `json:"source_store_mode"`
	SourceStoreStage                                                string `json:"source_store_stage"`
	SourceStoreAppendOnly                                           bool   `json:"source_store_append_only"`
	SourceStoreReadBack                                             bool   `json:"source_store_read_back"`
	SourceStoreReceiptPersisted                                     bool   `json:"source_store_receipt_persisted"`
	SourceStoreReceiptVerified                                      bool   `json:"source_store_receipt_verified"`
	SourceStoreAdmissionRequired                                    bool   `json:"source_store_admission_required"`
	SourceStoreShadowOnly                                           bool   `json:"source_store_shadow_only"`
	SourceStoreGraftAllowed                                         bool   `json:"source_store_graft_allowed"`
	SourceStoreDryRunOnly                                           bool   `json:"source_store_dry_run_only"`
	SourceStoreLiveReady                                            bool   `json:"source_store_live_ready"`
	SourceStoreRawDreamTextAllowed                                  bool   `json:"source_store_raw_dream_text_allowed"`
	SourceStoreRawDreamTextObserved                                 bool   `json:"source_store_raw_dream_text_observed"`
	SourceStoreRawDreamTextForwarded                                bool   `json:"source_store_raw_dream_text_forwarded"`
	SourceStoreJanusSurfaceAllowed                                  bool   `json:"source_store_janus_surface_allowed"`
	SourceStoreCoocLearningAllowed                                  bool   `json:"source_store_cooc_learning_allowed"`
	SourceStoreDeltaHarvestAllowed                                  bool   `json:"source_store_delta_harvest_allowed"`
	SourceStoreBodyMutationAllowed                                  bool   `json:"source_store_body_mutation_allowed"`
	SourceStoreRollbackRequired                                     bool   `json:"source_store_rollback_required"`
	SourceWeightedAdmissionResonanceGraftCandidateID                string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady             bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateCausalID          string `json:"source_weighted_admission_resonance_graft_candidate_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateHash              string `json:"source_weighted_admission_resonance_graft_candidate_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateReadBackHash      string `json:"source_weighted_admission_resonance_graft_candidate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftGateID                     string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                  bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftGateCausalID               string `json:"source_weighted_admission_resonance_graft_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftGateHash                   string `json:"source_weighted_admission_resonance_graft_gate_hash"`
	SourceWeightedAdmissionResonanceGraftGateReadBackHash           string `json:"source_weighted_admission_resonance_graft_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftPreflightID                string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady             bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                 string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady              bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                   string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                      string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                   bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                     bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                           bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                              bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                        bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                        bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                               bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                             bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                    bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                         bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                          bool   `json:"source_authority_granted"`
	AuthorityGranted                                                bool   `json:"authority_granted"`
	ContractsReady                                                  bool   `json:"contracts_ready"`
	WriteAllowed                                                    bool   `json:"write_allowed"`
	AdmissionAllowed                                                bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                            bool   `json:"live_admission_enabled"`
	MutatesState                                                    bool   `json:"mutates_state"`
	BodyTarget                                                      string `json:"body_target"`
	Passed                                                          bool   `json:"passed"`
	Reason                                                          string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader RESONANCE_GRAFT_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT")
	}
	storePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader output path missing")
	}
	store, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportForAssert(storePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportError(store, root); err != nil {
		return err
	}
	reader := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport{
		Schema:       admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema,
		Status:       "shadow_graft_candidate_store_read_back_dry_run",
		Target:       "resonance",
		TargetKind:   "weighted_internal_world_shadow_graft_candidate_store_reader",
		TargetMode:   "read_only_replay_dry_run",
		Action:       "read_weighted_resonance_shadow_graft_candidate_store_dry_run",
		ReceiptShape: "weighted_resonance_shadow_graft_candidate_store_reader_receipt",
		ReaderKind:   "shadow_graft_candidate_store_reader",
		ReaderMode:   "read_only_replay",
		ReaderStage:  "pre_live_graft_candidate_store_reader",
		WeightedAdmissionResonanceGraftCandidateStoreReaderReady: true,
		WeightedAdmissionResonanceGraftCandidateStoreConsumed:    true,
		WeightedAdmissionResonanceGraftCandidateStoreRequired:    true,
		NextStepBlockedWithoutResonanceGraftCandidateStoreReader: true,
		StoreVerified:          true,
		CandidateVerified:      store.CandidateVerified,
		GateVerified:           store.GateVerified,
		PreflightVerified:      store.PreflightVerified,
		BoundaryVerified:       store.BoundaryVerified,
		ObservationVerified:    store.ObservationVerified,
		ReceiverVerified:       store.ReceiverVerified,
		IntentVerified:         store.IntentVerified,
		FinalGateVerified:      store.FinalGateVerified,
		SealVerified:           store.SealVerified,
		PermitVerified:         store.PermitVerified,
		AuthorityVerified:      store.AuthorityVerified,
		StoreHashVerified:      true,
		StoreReadBackVerified:  true,
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
		SourceAppendOnly:       store.AppendOnly,
		SourceReadBack:         store.ReadBack,
		SourceReceiptPersisted: store.ReceiptPersisted,
		SourceReceiptVerified:  store.ReceiptVerified,
		SourceSchema:           store.Schema,
		SourceStatus:           store.Status,
		SourceTarget:           store.Target,
		SourceReport:           storePath,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:           store.WeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:        store.WeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID:     store.CausalID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreHash:         store.StoreHash,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash: store.ReadBackHash,
		SourceStoreAction:                                          store.Action,
		SourceStoreReceiptShape:                                    store.ReceiptShape,
		SourceStoreKind:                                            store.StoreKind,
		SourceStoreMode:                                            store.StoreMode,
		SourceStoreStage:                                           store.StoreStage,
		SourceStoreAppendOnly:                                      store.AppendOnly,
		SourceStoreReadBack:                                        store.ReadBack,
		SourceStoreReceiptPersisted:                                store.ReceiptPersisted,
		SourceStoreReceiptVerified:                                 store.ReceiptVerified,
		SourceStoreAdmissionRequired:                               store.AdmissionRequired,
		SourceStoreShadowOnly:                                      store.ShadowOnly,
		SourceStoreGraftAllowed:                                    store.GraftAllowed,
		SourceStoreDryRunOnly:                                      store.DryRunOnly,
		SourceStoreLiveReady:                                       store.LiveReady,
		SourceStoreRawDreamTextAllowed:                             store.RawDreamTextAllowed,
		SourceStoreRawDreamTextObserved:                            store.RawDreamTextObserved,
		SourceStoreRawDreamTextForwarded:                           store.RawDreamTextForwarded,
		SourceStoreJanusSurfaceAllowed:                             store.JanusSurfaceAllowed,
		SourceStoreCoocLearningAllowed:                             store.CoocLearningAllowed,
		SourceStoreDeltaHarvestAllowed:                             store.DeltaHarvestAllowed,
		SourceStoreBodyMutationAllowed:                             store.BodyMutationAllowed,
		SourceStoreRollbackRequired:                                store.RollbackRequired,
		SourceWeightedAdmissionResonanceGraftCandidateID:           store.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:        store.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftCandidateCausalID:     store.SourceWeightedAdmissionResonanceGraftCandidateCausalID,
		SourceWeightedAdmissionResonanceGraftCandidateHash:         store.SourceWeightedAdmissionResonanceGraftCandidateHash,
		SourceWeightedAdmissionResonanceGraftCandidateReadBackHash: store.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash,
		SourceWeightedAdmissionResonanceGraftGateID:                store.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:             store.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftGateCausalID:          store.SourceWeightedAdmissionResonanceGraftGateCausalID,
		SourceWeightedAdmissionResonanceGraftGateHash:              store.SourceWeightedAdmissionResonanceGraftGateHash,
		SourceWeightedAdmissionResonanceGraftGateReadBackHash:      store.SourceWeightedAdmissionResonanceGraftGateReadBackHash,
		SourceWeightedAdmissionResonanceGraftPreflightID:           store.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:        store.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:            store.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:         store.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:              store.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:           store.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                 store.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:              store.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                store.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                      store.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                           store.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                         store.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                   store.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                   store.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                          store.BodySmokeWeighted,
		NanoDirectRunner:                                           store.NanoDirectRunner,
		NanoDirectFinalGate:                                        store.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                               store.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                    store.BoundaryReportFullChain,
		SourceAuthorityGranted:                                     store.SourceAuthorityGranted,
		AuthorityGranted:                                           false,
		ContractsReady:                                             false,
		WriteAllowed:                                               false,
		AdmissionAllowed:                                           false,
		LiveAdmissionEnabled:                                       false,
		MutatesState:                                               false,
		BodyTarget:                                                 "none",
		Passed:                                                     true,
		Reason:                                                     "weighted resonance shadow graft candidate store read back without body mutation",
	}
	reader.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID(reader)
	reader.ReaderHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderHash(reader)
	reader.ReplayHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash(reader)
	reader.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReadBackHash(reader)
	reader.WeightedAdmissionResonanceGraftCandidateStoreReaderID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderID(reader)
	if reader.CausalID == "" ||
		reader.ReaderHash == "" ||
		reader.ReplayHash == "" ||
		reader.ReadBackHash == "" ||
		reader.WeightedAdmissionResonanceGraftCandidateStoreReaderID == "" ||
		reader.ReaderHash == reader.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store reader read-back proof failed")
	}
	raw, err := json.MarshalIndent(reader, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate store reader marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate store reader write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-candidate-store-reader] pass: resonance_graft_candidate_store_reader_report=%s resonance_graft_candidate_store_report=%s\n", outputPath, storePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft candidate store reader schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema {
		return fmt.Errorf("weighted admission resonance graft candidate store reader schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema)
	}
	if report.Status != "shadow_graft_candidate_store_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader status mismatch: got %q want %q", report.Status, "shadow_graft_candidate_store_read_back_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_candidate_store_reader" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_candidate_store_reader")
	}
	if report.TargetMode != "read_only_replay_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader target_mode mismatch: got %q want %q", report.TargetMode, "read_only_replay_dry_run")
	}
	if report.Action != "read_weighted_resonance_shadow_graft_candidate_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader action mismatch: got %q want %q", report.Action, "read_weighted_resonance_shadow_graft_candidate_store_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_candidate_store_reader_receipt" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_candidate_store_reader_receipt")
	}
	if report.ReaderKind != "shadow_graft_candidate_store_reader" ||
		report.ReaderMode != "read_only_replay" ||
		report.ReaderStage != "pre_live_graft_candidate_store_reader" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_candidate_store_reader_ready", report.WeightedAdmissionResonanceGraftCandidateStoreReaderReady},
		{"weighted_admission_resonance_graft_candidate_store_consumed", report.WeightedAdmissionResonanceGraftCandidateStoreConsumed},
		{"weighted_admission_resonance_graft_candidate_store_required", report.WeightedAdmissionResonanceGraftCandidateStoreRequired},
		{"next_step_blocked_without_resonance_graft_candidate_store_reader", report.NextStepBlockedWithoutResonanceGraftCandidateStoreReader},
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
		{"store_hash_verified", report.StoreHashVerified},
		{"store_read_back_verified", report.StoreReadBackVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_append_only", report.SourceAppendOnly},
		{"source_read_back", report.SourceReadBack},
		{"source_receipt_persisted", report.SourceReceiptPersisted},
		{"source_receipt_verified", report.SourceReceiptVerified},
		{"source_weighted_admission_resonance_graft_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReady},
		{"source_store_append_only", report.SourceStoreAppendOnly},
		{"source_store_read_back", report.SourceStoreReadBack},
		{"source_store_receipt_persisted", report.SourceStoreReceiptPersisted},
		{"source_store_receipt_verified", report.SourceStoreReceiptVerified},
		{"source_store_admission_required", report.SourceStoreAdmissionRequired},
		{"source_store_shadow_only", report.SourceStoreShadowOnly},
		{"source_store_dry_run_only", report.SourceStoreDryRunOnly},
		{"source_store_live_ready", report.SourceStoreLiveReady},
		{"source_store_rollback_required", report.SourceStoreRollbackRequired},
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
			return fmt.Errorf("weighted admission resonance graft candidate store reader %s not ready", required.name)
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
		{"source_store_graft_allowed", report.SourceStoreGraftAllowed},
		{"source_store_raw_dream_text_allowed", report.SourceStoreRawDreamTextAllowed},
		{"source_store_raw_dream_text_observed", report.SourceStoreRawDreamTextObserved},
		{"source_store_raw_dream_text_forwarded", report.SourceStoreRawDreamTextForwarded},
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
			return fmt.Errorf("weighted admission resonance graft candidate store reader opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_candidate_store_reader_id", report.WeightedAdmissionResonanceGraftCandidateStoreReaderID},
		{"causal_id", report.CausalID},
		{"reader_hash", report.ReaderHash},
		{"replay_hash", report.ReplayHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
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
			return fmt.Errorf("weighted admission resonance graft candidate store reader %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema)
	}
	if report.SourceStatus != "shadow_graft_candidate_stored_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_candidate_stored_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceStoreAction != "store_weighted_resonance_shadow_graft_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source_store_action mismatch: got %q want %q", report.SourceStoreAction, "store_weighted_resonance_shadow_graft_candidate_dry_run")
	}
	if report.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source_store_receipt_shape mismatch: got %q want %q", report.SourceStoreReceiptShape, "weighted_resonance_shadow_graft_candidate_store_receipt")
	}
	if report.SourceStoreKind != "shadow_graft_candidate_store" ||
		report.SourceStoreMode != "append_only_read_back_store" ||
		report.SourceStoreStage != "pre_live_graft_candidate_store" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source store shape mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-candidate-store-reader-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader causal prefix mismatch")
	}
	if !strings.HasPrefix(report.ReaderHash, "weighted-resonance-graft-candidate-store-reader-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReplayHash, "weighted-resonance-graft-candidate-store-reader-replay-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader replay prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-candidate-store-reader-read-") ||
		report.ReaderHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store reader read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID, "weighted-resonance-graft-candidate-store-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source store causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreHash, "weighted-resonance-graft-candidate-store-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source store hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash, "weighted-resonance-graft-candidate-store-read-") ||
		report.SourceWeightedAdmissionResonanceGraftCandidateStoreHash == report.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source store read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateCausalID, "weighted-resonance-graft-candidate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateHash, "weighted-resonance-graft-candidate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash, "weighted-resonance-graft-candidate-read-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source candidate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateCausalID, "weighted-resonance-graft-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateHash, "weighted-resonance-graft-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateReadBackHash, "weighted-resonance-graft-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft candidate store reader source receiver id prefix mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store reader causal_id mismatch")
	}
	if report.ReaderHash == "" || report.ReaderHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store reader reader_hash mismatch")
	}
	if report.ReplayHash == "" || report.ReplayHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store reader replay_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store reader read_back_hash mismatch")
	}
	if report.ReaderHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store reader read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftCandidateStoreReaderID == "" || report.WeightedAdmissionResonanceGraftCandidateStoreReaderID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store reader id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft candidate store read back without body mutation" {
		return fmt.Errorf("weighted admission resonance graft candidate store reader reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID(reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		SourceStoreID       string `json:"source_store_id"`
		SourceStoreReadBack string `json:"source_store_read_back_hash"`
		SourceCandidateID   string `json:"source_candidate_id"`
		SourceGateID        string `json:"source_gate_id"`
		SourceObservationID string `json:"source_observation_id"`
		Target              string `json:"target"`
		ReaderKind          string `json:"reader_kind"`
		ReaderStage         string `json:"reader_stage"`
	}{
		SourceStoreID:       reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceStoreReadBack: reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		SourceCandidateID:   reader.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:        reader.SourceWeightedAdmissionResonanceGraftGateID,
		SourceObservationID: reader.SourceWeightedAdmissionResonanceObservationID,
		Target:              reader.Target,
		ReaderKind:          reader.ReaderKind,
		ReaderStage:         reader.ReaderStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-reader-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport) string {
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
		GraftAllowed   bool   `json:"graft_allowed"`
		BodyMutation   bool   `json:"body_mutation"`
		LiveAdmission  bool   `json:"live_admission"`
	}{
		CausalID:       reader.CausalID,
		SourceStoreID:  reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceHash:     reader.SourceWeightedAdmissionResonanceGraftCandidateStoreHash,
		SourceReadBack: reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		ReaderMode:     reader.ReaderMode,
		ReceiptShape:   reader.ReceiptShape,
		ReadOnly:       reader.ReadOnly,
		ReplayOnly:     reader.ReplayOnly,
		StoreVerified:  reader.StoreVerified,
		SourceAppend:   reader.SourceAppendOnly,
		SourceRead:     reader.SourceReadBack,
		SourceVerified: reader.SourceReceiptVerified,
		GraftAllowed:   reader.GraftAllowed,
		BodyMutation:   reader.BodyMutationAllowed,
		LiveAdmission:  reader.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-reader-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		ReaderHash      string `json:"reader_hash"`
		SourceStoreID   string `json:"source_store_id"`
		SourceStoreHash string `json:"source_store_hash"`
		SourceCandidate string `json:"source_candidate_id"`
		ReadOnly        bool   `json:"read_only"`
		ReplayOnly      bool   `json:"replay_only"`
		ReaderReady     bool   `json:"reader_ready"`
	}{
		ReaderHash:      reader.ReaderHash,
		SourceStoreID:   reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceStoreHash: reader.SourceWeightedAdmissionResonanceGraftCandidateStoreHash,
		SourceCandidate: reader.SourceWeightedAdmissionResonanceGraftCandidateID,
		ReadOnly:        reader.ReadOnly,
		ReplayOnly:      reader.ReplayOnly,
		ReaderReady:     reader.WeightedAdmissionResonanceGraftCandidateStoreReaderReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-reader-replay-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReadBackHash(reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		ReaderHash      string `json:"reader_hash"`
		ReplayHash      string `json:"replay_hash"`
		SourceStoreRead string `json:"source_store_read_back_hash"`
		ReaderReady     bool   `json:"reader_ready"`
		ReadOnly        bool   `json:"read_only"`
		ReplayOnly      bool   `json:"replay_only"`
		WriteAllowed    bool   `json:"write_allowed"`
		MutatesState    bool   `json:"mutates_state"`
		AdmissionOpen   bool   `json:"admission_open"`
	}{
		ReaderHash:      reader.ReaderHash,
		ReplayHash:      reader.ReplayHash,
		SourceStoreRead: reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash,
		ReaderReady:     reader.WeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		ReadOnly:        reader.ReadOnly,
		ReplayOnly:      reader.ReplayOnly,
		WriteAllowed:    reader.WriteAllowed,
		MutatesState:    reader.MutatesState,
		AdmissionOpen:   reader.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-reader-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderID(reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceStoreID          string `json:"source_store_id"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		SourceReceiverID       string `json:"source_receiver_id"`
		CausalID               string `json:"causal_id"`
		ReaderHash             string `json:"reader_hash"`
		ReplayHash             string `json:"replay_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReceiptShape           string `json:"receipt_shape"`
		ReaderKind             string `json:"reader_kind"`
		ReaderMode             string `json:"reader_mode"`
		ReaderStage            string `json:"reader_stage"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		SourceAppendOnly       bool   `json:"source_append_only"`
		SourceReadBack         bool   `json:"source_read_back"`
		SourceReceiptPersisted bool   `json:"source_receipt_persisted"`
		SourceReceiptVerified  bool   `json:"source_receipt_verified"`
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
		StoreHashVerified      bool   `json:"store_hash_verified"`
		StoreReadBackVerified  bool   `json:"store_read_back_verified"`
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
		LiveReady              bool   `json:"live_ready"`
		ContractsReady         bool   `json:"contracts_ready"`
		BodyTarget             string `json:"body_target"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_candidate_store_reader"`
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
		Schema:                 reader.Schema,
		Status:                 reader.Status,
		Action:                 reader.Action,
		SourceReport:           reader.SourceReport,
		SourceStoreID:          reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:      reader.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:           reader.SourceWeightedAdmissionResonanceGraftGateID,
		SourceBoundaryID:       reader.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:    reader.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:       reader.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:               reader.CausalID,
		ReaderHash:             reader.ReaderHash,
		ReplayHash:             reader.ReplayHash,
		ReadBackHash:           reader.ReadBackHash,
		Ready:                  reader.WeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		ReceiptShape:           reader.ReceiptShape,
		ReaderKind:             reader.ReaderKind,
		ReaderMode:             reader.ReaderMode,
		ReaderStage:            reader.ReaderStage,
		ReadOnly:               reader.ReadOnly,
		ReplayOnly:             reader.ReplayOnly,
		SourceAppendOnly:       reader.SourceAppendOnly,
		SourceReadBack:         reader.SourceReadBack,
		SourceReceiptPersisted: reader.SourceReceiptPersisted,
		SourceReceiptVerified:  reader.SourceReceiptVerified,
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
		StoreHashVerified:      reader.StoreHashVerified,
		StoreReadBackVerified:  reader.StoreReadBackVerified,
		AdmissionRequired:      reader.AdmissionRequired,
		ShadowOnly:             reader.ShadowOnly,
		GraftAllowed:           reader.GraftAllowed,
		DryRunOnly:             reader.DryRunOnly,
		RawDreamTextAllowed:    reader.RawDreamTextAllowed,
		JanusSurfaceAllowed:    reader.JanusSurfaceAllowed,
		CoocLearningAllowed:    reader.CoocLearningAllowed,
		DeltaHarvestAllowed:    reader.DeltaHarvestAllowed,
		BodyMutationAllowed:    reader.BodyMutationAllowed,
		RollbackRequired:       reader.RollbackRequired,
		LiveReady:              reader.LiveReady,
		ContractsReady:         reader.ContractsReady,
		BodyTarget:             reader.BodyTarget,
		WriteAllowed:           reader.WriteAllowed,
		AdmissionAllowed:       reader.AdmissionAllowed,
		LiveAdmissionEnabled:   reader.LiveAdmissionEnabled,
		MutatesState:           reader.MutatesState,
		NextStepBlockedWithout: reader.NextStepBlockedWithoutResonanceGraftCandidateStoreReader,
		SourceStoreReady:       reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:   reader.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:        reader.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:   reader.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:    reader.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady: reader.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:    reader.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:      reader.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:   reader.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:        reader.SourceWeightedAdmissionSealReady,
		SourcePermitReady:      reader.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:    reader.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:  reader.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-reader-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store reader path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft candidate store reader not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store reader not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store reader JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store reader decode failed: %w", err)
	}
	return report, root, nil
}
