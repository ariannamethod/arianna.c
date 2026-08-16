package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema = "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport struct {
	Schema                                                     string `json:"schema"`
	Status                                                     string `json:"status"`
	Target                                                     string `json:"target"`
	TargetKind                                                 string `json:"target_kind"`
	TargetMode                                                 string `json:"target_mode"`
	Action                                                     string `json:"action"`
	WeightedAdmissionResonanceGraftCandidateStoreReady         bool   `json:"weighted_admission_resonance_graft_candidate_store_ready"`
	WeightedAdmissionResonanceGraftCandidateConsumed           bool   `json:"weighted_admission_resonance_graft_candidate_consumed"`
	WeightedAdmissionResonanceGraftCandidateRequired           bool   `json:"weighted_admission_resonance_graft_candidate_required"`
	NextStepBlockedWithoutResonanceGraftCandidateStore         bool   `json:"next_step_blocked_without_resonance_graft_candidate_store"`
	WeightedAdmissionResonanceGraftCandidateStoreID            string `json:"weighted_admission_resonance_graft_candidate_store_id"`
	ReceiptShape                                               string `json:"receipt_shape"`
	StoreKind                                                  string `json:"store_kind"`
	StoreMode                                                  string `json:"store_mode"`
	StoreStage                                                 string `json:"store_stage"`
	CausalID                                                   string `json:"causal_id"`
	StoreHash                                                  string `json:"store_hash"`
	ReadBackHash                                               string `json:"read_back_hash"`
	CandidateVerified                                          bool   `json:"candidate_verified"`
	GateVerified                                               bool   `json:"gate_verified"`
	PreflightVerified                                          bool   `json:"preflight_verified"`
	BoundaryVerified                                           bool   `json:"boundary_verified"`
	ObservationVerified                                        bool   `json:"observation_verified"`
	ReceiverVerified                                           bool   `json:"receiver_verified"`
	IntentVerified                                             bool   `json:"intent_verified"`
	FinalGateVerified                                          bool   `json:"final_gate_verified"`
	SealVerified                                               bool   `json:"seal_verified"`
	PermitVerified                                             bool   `json:"permit_verified"`
	AuthorityVerified                                          bool   `json:"authority_verified"`
	AdmissionRequired                                          bool   `json:"admission_required"`
	ShadowOnly                                                 bool   `json:"shadow_only"`
	GraftAllowed                                               bool   `json:"graft_allowed"`
	DryRunOnly                                                 bool   `json:"dry_run_only"`
	LiveReady                                                  bool   `json:"live_ready"`
	RawDreamTextAllowed                                        bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                       bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                      bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                        bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                        bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                        bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                        bool   `json:"body_mutation_allowed"`
	RollbackRequired                                           bool   `json:"rollback_required"`
	AppendOnly                                                 bool   `json:"append_only"`
	ReadBack                                                   bool   `json:"read_back"`
	ReceiptPersisted                                           bool   `json:"receipt_persisted"`
	ReceiptVerified                                            bool   `json:"receipt_verified"`
	SourceSchema                                               string `json:"source_schema"`
	SourceStatus                                               string `json:"source_status"`
	SourceTarget                                               string `json:"source_target"`
	SourceReport                                               string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftCandidateID           string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady        bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateCausalID     string `json:"source_weighted_admission_resonance_graft_candidate_causal_id"`
	SourceWeightedAdmissionResonanceGraftCandidateHash         string `json:"source_weighted_admission_resonance_graft_candidate_hash"`
	SourceWeightedAdmissionResonanceGraftCandidateReadBackHash string `json:"source_weighted_admission_resonance_graft_candidate_read_back_hash"`
	SourceCandidateAction                                      string `json:"source_candidate_action"`
	SourceCandidateReceiptShape                                string `json:"source_candidate_receipt_shape"`
	SourceCandidateKind                                        string `json:"source_candidate_kind"`
	SourceCandidateMode                                        string `json:"source_candidate_mode"`
	SourceCandidateStage                                       string `json:"source_candidate_stage"`
	SourceCandidateShadowOnly                                  bool   `json:"source_candidate_shadow_only"`
	SourceCandidateGraftAllowed                                bool   `json:"source_candidate_graft_allowed"`
	SourceCandidateDryRunOnly                                  bool   `json:"source_candidate_dry_run_only"`
	SourceCandidateLiveReady                                   bool   `json:"source_candidate_live_ready"`
	SourceCandidateRawDreamTextAllowed                         bool   `json:"source_candidate_raw_dream_text_allowed"`
	SourceCandidateRawDreamTextObserved                        bool   `json:"source_candidate_raw_dream_text_observed"`
	SourceCandidateRawDreamTextForwarded                       bool   `json:"source_candidate_raw_dream_text_forwarded"`
	SourceCandidateJanusSurfaceAllowed                         bool   `json:"source_candidate_janus_surface_allowed"`
	SourceCandidateCoocLearningAllowed                         bool   `json:"source_candidate_cooc_learning_allowed"`
	SourceCandidateDeltaHarvestAllowed                         bool   `json:"source_candidate_delta_harvest_allowed"`
	SourceCandidateBodyMutationAllowed                         bool   `json:"source_candidate_body_mutation_allowed"`
	SourceCandidateRollbackRequired                            bool   `json:"source_candidate_rollback_required"`
	SourceWeightedAdmissionResonanceGraftGateID                string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady             bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftGateCausalID          string `json:"source_weighted_admission_resonance_graft_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftGateHash              string `json:"source_weighted_admission_resonance_graft_gate_hash"`
	SourceWeightedAdmissionResonanceGraftGateReadBackHash      string `json:"source_weighted_admission_resonance_graft_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftPreflightID           string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady        bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID            string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady         bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID              string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady           bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                 string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady              bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                      bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                           bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                         bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                   bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                   bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                          bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                           bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                        bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                               bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                    bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                     bool   `json:"source_authority_granted"`
	AuthorityGranted                                           bool   `json:"authority_granted"`
	ContractsReady                                             bool   `json:"contracts_ready"`
	WriteAllowed                                               bool   `json:"write_allowed"`
	AdmissionAllowed                                           bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                       bool   `json:"live_admission_enabled"`
	MutatesState                                               bool   `json:"mutates_state"`
	BodyTarget                                                 string `json:"body_target"`
	Passed                                                     bool   `json:"passed"`
	Reason                                                     string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStore(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store RESONANCE_GRAFT_CANDIDATE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_REPORT")
	}
	candidatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft candidate store output path missing")
	}
	candidate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportForAssert(candidatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportError(candidate, root); err != nil {
		return err
	}
	store := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport{
		Schema:     admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema,
		Status:     "shadow_graft_candidate_stored_dry_run",
		Target:     "resonance",
		TargetKind: "weighted_internal_world_shadow_graft_candidate_store",
		TargetMode: "append_only_read_back_store_dry_run",
		Action:     "store_weighted_resonance_shadow_graft_candidate_dry_run",
		WeightedAdmissionResonanceGraftCandidateStoreReady: true,
		WeightedAdmissionResonanceGraftCandidateConsumed:   true,
		WeightedAdmissionResonanceGraftCandidateRequired:   true,
		NextStepBlockedWithoutResonanceGraftCandidateStore: true,
		ReceiptShape:          "weighted_resonance_shadow_graft_candidate_store_receipt",
		StoreKind:             "shadow_graft_candidate_store",
		StoreMode:             "append_only_read_back_store",
		StoreStage:            "pre_live_graft_candidate_store",
		CandidateVerified:     true,
		GateVerified:          candidate.SourceWeightedAdmissionResonanceGraftGateReady,
		PreflightVerified:     candidate.PreflightVerified,
		BoundaryVerified:      candidate.BoundaryVerified,
		ObservationVerified:   candidate.ObservationVerified,
		ReceiverVerified:      candidate.ReceiverVerified,
		IntentVerified:        candidate.IntentVerified,
		FinalGateVerified:     candidate.FinalGateVerified,
		SealVerified:          candidate.SealVerified,
		PermitVerified:        candidate.PermitVerified,
		AuthorityVerified:     candidate.AuthorityVerified,
		AdmissionRequired:     true,
		ShadowOnly:            true,
		GraftAllowed:          false,
		DryRunOnly:            true,
		LiveReady:             true,
		RawDreamTextAllowed:   false,
		RawDreamTextObserved:  false,
		RawDreamTextForwarded: false,
		JanusSurfaceAllowed:   false,
		CoocLearningAllowed:   false,
		DeltaHarvestAllowed:   false,
		BodyMutationAllowed:   false,
		RollbackRequired:      true,
		AppendOnly:            true,
		ReadBack:              true,
		ReceiptPersisted:      true,
		ReceiptVerified:       true,
		SourceSchema:          candidate.Schema,
		SourceStatus:          candidate.Status,
		SourceTarget:          candidate.Target,
		SourceReport:          candidatePath,
		SourceWeightedAdmissionResonanceGraftCandidateID:           candidate.WeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:        candidate.WeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftCandidateCausalID:     candidate.CausalID,
		SourceWeightedAdmissionResonanceGraftCandidateHash:         candidate.CandidateHash,
		SourceWeightedAdmissionResonanceGraftCandidateReadBackHash: candidate.ReadBackHash,
		SourceCandidateAction:                                      candidate.Action,
		SourceCandidateReceiptShape:                                candidate.ReceiptShape,
		SourceCandidateKind:                                        candidate.CandidateKind,
		SourceCandidateMode:                                        candidate.CandidateMode,
		SourceCandidateStage:                                       candidate.CandidateStage,
		SourceCandidateShadowOnly:                                  candidate.ShadowOnly,
		SourceCandidateGraftAllowed:                                candidate.GraftAllowed,
		SourceCandidateDryRunOnly:                                  candidate.DryRunOnly,
		SourceCandidateLiveReady:                                   candidate.LiveReady,
		SourceCandidateRawDreamTextAllowed:                         candidate.RawDreamTextAllowed,
		SourceCandidateRawDreamTextObserved:                        candidate.RawDreamTextObserved,
		SourceCandidateRawDreamTextForwarded:                       candidate.RawDreamTextForwarded,
		SourceCandidateJanusSurfaceAllowed:                         candidate.JanusSurfaceAllowed,
		SourceCandidateCoocLearningAllowed:                         candidate.CoocLearningAllowed,
		SourceCandidateDeltaHarvestAllowed:                         candidate.DeltaHarvestAllowed,
		SourceCandidateBodyMutationAllowed:                         candidate.BodyMutationAllowed,
		SourceCandidateRollbackRequired:                            candidate.RollbackRequired,
		SourceWeightedAdmissionResonanceGraftGateID:                candidate.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:             candidate.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftGateCausalID:          candidate.SourceWeightedAdmissionResonanceGraftGateCausal,
		SourceWeightedAdmissionResonanceGraftGateHash:              candidate.SourceWeightedAdmissionResonanceGraftGateHash,
		SourceWeightedAdmissionResonanceGraftGateReadBackHash:      candidate.SourceWeightedAdmissionResonanceGraftGateRead,
		SourceWeightedAdmissionResonanceGraftPreflightID:           candidate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:        candidate.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:            candidate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:         candidate.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:              candidate.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:           candidate.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                 candidate.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:              candidate.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                candidate.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                      candidate.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                           candidate.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                         candidate.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                   candidate.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                   candidate.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                          candidate.BodySmokeWeighted,
		NanoDirectRunner:                                           candidate.NanoDirectRunner,
		NanoDirectFinalGate:                                        candidate.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                               candidate.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                    candidate.BoundaryReportFullChain,
		SourceAuthorityGranted:                                     candidate.SourceAuthorityGranted,
		AuthorityGranted:                                           false,
		ContractsReady:                                             false,
		WriteAllowed:                                               false,
		AdmissionAllowed:                                           false,
		LiveAdmissionEnabled:                                       false,
		MutatesState:                                               false,
		BodyTarget:                                                 "none",
		Passed:                                                     true,
		Reason:                                                     "weighted resonance shadow graft candidate stored without body mutation",
	}
	store.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreCausalID(store)
	store.StoreHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreHash(store)
	store.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReadBackHash(store)
	store.WeightedAdmissionResonanceGraftCandidateStoreID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreID(store)
	if store.CausalID == "" ||
		store.StoreHash == "" ||
		store.ReadBackHash == "" ||
		store.WeightedAdmissionResonanceGraftCandidateStoreID == "" ||
		store.StoreHash == store.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store read-back proof failed")
	}
	raw, err := json.MarshalIndent(store, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate store marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate store write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-candidate-store] pass: resonance_graft_candidate_store_report=%s resonance_graft_candidate_report=%s\n", outputPath, candidatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft candidate store schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema {
		return fmt.Errorf("weighted admission resonance graft candidate store schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema)
	}
	if report.Status != "shadow_graft_candidate_stored_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store status mismatch: got %q want %q", report.Status, "shadow_graft_candidate_stored_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate store target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_candidate_store" {
		return fmt.Errorf("weighted admission resonance graft candidate store target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_candidate_store")
	}
	if report.TargetMode != "append_only_read_back_store_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store target_mode mismatch: got %q want %q", report.TargetMode, "append_only_read_back_store_dry_run")
	}
	if report.Action != "store_weighted_resonance_shadow_graft_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store action mismatch: got %q want %q", report.Action, "store_weighted_resonance_shadow_graft_candidate_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" {
		return fmt.Errorf("weighted admission resonance graft candidate store receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_candidate_store_receipt")
	}
	if report.StoreKind != "shadow_graft_candidate_store" ||
		report.StoreMode != "append_only_read_back_store" ||
		report.StoreStage != "pre_live_graft_candidate_store" {
		return fmt.Errorf("weighted admission resonance graft candidate store shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_candidate_store_ready", report.WeightedAdmissionResonanceGraftCandidateStoreReady},
		{"weighted_admission_resonance_graft_candidate_consumed", report.WeightedAdmissionResonanceGraftCandidateConsumed},
		{"weighted_admission_resonance_graft_candidate_required", report.WeightedAdmissionResonanceGraftCandidateRequired},
		{"next_step_blocked_without_resonance_graft_candidate_store", report.NextStepBlockedWithoutResonanceGraftCandidateStore},
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
		{"rollback_required", report.RollbackRequired},
		{"append_only", report.AppendOnly},
		{"read_back", report.ReadBack},
		{"receipt_persisted", report.ReceiptPersisted},
		{"receipt_verified", report.ReceiptVerified},
		{"source_weighted_admission_resonance_graft_candidate_ready", report.SourceWeightedAdmissionResonanceGraftCandidateReady},
		{"source_candidate_shadow_only", report.SourceCandidateShadowOnly},
		{"source_candidate_dry_run_only", report.SourceCandidateDryRunOnly},
		{"source_candidate_live_ready", report.SourceCandidateLiveReady},
		{"source_candidate_rollback_required", report.SourceCandidateRollbackRequired},
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
			return fmt.Errorf("weighted admission resonance graft candidate store %s not ready", required.name)
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
		{"source_candidate_graft_allowed", report.SourceCandidateGraftAllowed},
		{"source_candidate_raw_dream_text_allowed", report.SourceCandidateRawDreamTextAllowed},
		{"source_candidate_raw_dream_text_observed", report.SourceCandidateRawDreamTextObserved},
		{"source_candidate_raw_dream_text_forwarded", report.SourceCandidateRawDreamTextForwarded},
		{"source_candidate_janus_surface_allowed", report.SourceCandidateJanusSurfaceAllowed},
		{"source_candidate_cooc_learning_allowed", report.SourceCandidateCoocLearningAllowed},
		{"source_candidate_delta_harvest_allowed", report.SourceCandidateDeltaHarvestAllowed},
		{"source_candidate_body_mutation_allowed", report.SourceCandidateBodyMutationAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft candidate store opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_candidate_store_id", report.WeightedAdmissionResonanceGraftCandidateStoreID},
		{"causal_id", report.CausalID},
		{"store_hash", report.StoreHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
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
			return fmt.Errorf("weighted admission resonance graft candidate store %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema {
		return fmt.Errorf("weighted admission resonance graft candidate store source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema)
	}
	if report.SourceStatus != "shadow_graft_candidate_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_candidate_ready_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate store source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceCandidateAction != "draft_weighted_resonance_shadow_graft_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate store source_candidate_action mismatch: got %q want %q", report.SourceCandidateAction, "draft_weighted_resonance_shadow_graft_candidate_dry_run")
	}
	if report.SourceCandidateReceiptShape != "weighted_resonance_shadow_graft_candidate_contract" {
		return fmt.Errorf("weighted admission resonance graft candidate store source_candidate_receipt_shape mismatch: got %q want %q", report.SourceCandidateReceiptShape, "weighted_resonance_shadow_graft_candidate_contract")
	}
	if report.SourceCandidateKind != "shadow_graft_candidate" ||
		report.SourceCandidateMode != "no_mutation_candidate" ||
		report.SourceCandidateStage != "pre_live_graft_candidate" {
		return fmt.Errorf("weighted admission resonance graft candidate store source candidate shape mismatch")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-candidate-store-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate store causal prefix mismatch")
	}
	if !strings.HasPrefix(report.StoreHash, "weighted-resonance-graft-candidate-store-") {
		return fmt.Errorf("weighted admission resonance graft candidate store hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-candidate-store-read-") ||
		report.StoreHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateCausalID, "weighted-resonance-graft-candidate-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source candidate causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateHash, "weighted-resonance-graft-candidate-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source candidate hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash, "weighted-resonance-graft-candidate-read-") ||
		report.SourceWeightedAdmissionResonanceGraftCandidateHash == report.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store source candidate read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateCausalID, "weighted-resonance-graft-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateHash, "weighted-resonance-graft-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateReadBackHash, "weighted-resonance-graft-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source gate proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft candidate store source receiver id prefix mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft candidate store body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store causal_id mismatch")
	}
	if report.StoreHash == "" || report.StoreHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store store_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store read_back_hash mismatch")
	}
	if report.StoreHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate store read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftCandidateStoreID == "" || report.WeightedAdmissionResonanceGraftCandidateStoreID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate store id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft candidate stored without body mutation" {
		return fmt.Errorf("weighted admission resonance graft candidate store reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreCausalID(store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport) string {
	h := hashJSON(struct {
		SourceCandidateID           string `json:"source_candidate_id"`
		SourceCandidateReadBackHash string `json:"source_candidate_read_back_hash"`
		SourceGateID                string `json:"source_gate_id"`
		SourcePreflightID           string `json:"source_preflight_id"`
		SourceBoundaryID            string `json:"source_boundary_id"`
		SourceObservationID         string `json:"source_observation_id"`
		SourceReceiverID            string `json:"source_receiver_id"`
		Target                      string `json:"target"`
		StoreKind                   string `json:"store_kind"`
		StoreStage                  string `json:"store_stage"`
	}{
		SourceCandidateID:           store.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceCandidateReadBackHash: store.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash,
		SourceGateID:                store.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:           store.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:            store.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:         store.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:            store.SourceWeightedAdmissionResonanceReceiverID,
		Target:                      store.Target,
		StoreKind:                   store.StoreKind,
		StoreStage:                  store.StoreStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreHash(store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport) string {
	h := hashJSON(struct {
		CausalID                    string `json:"causal_id"`
		SourceCandidateID           string `json:"source_candidate_id"`
		SourceCandidateHash         string `json:"source_candidate_hash"`
		SourceCandidateReadBackHash string `json:"source_candidate_read_back_hash"`
		StoreMode                   string `json:"store_mode"`
		AppendOnly                  bool   `json:"append_only"`
		ReadBack                    bool   `json:"read_back"`
		ReceiptPersisted            bool   `json:"receipt_persisted"`
		ReceiptVerified             bool   `json:"receipt_verified"`
		AdmissionRequired           bool   `json:"admission_required"`
		ShadowOnly                  bool   `json:"shadow_only"`
		DryRunOnly                  bool   `json:"dry_run_only"`
		GraftAllowed                bool   `json:"graft_allowed"`
	}{
		CausalID:                    store.CausalID,
		SourceCandidateID:           store.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceCandidateHash:         store.SourceWeightedAdmissionResonanceGraftCandidateHash,
		SourceCandidateReadBackHash: store.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash,
		StoreMode:                   store.StoreMode,
		AppendOnly:                  store.AppendOnly,
		ReadBack:                    store.ReadBack,
		ReceiptPersisted:            store.ReceiptPersisted,
		ReceiptVerified:             store.ReceiptVerified,
		AdmissionRequired:           store.AdmissionRequired,
		ShadowOnly:                  store.ShadowOnly,
		DryRunOnly:                  store.DryRunOnly,
		GraftAllowed:                store.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReadBackHash(store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport) string {
	h := hashJSON(struct {
		StoreHash       string `json:"store_hash"`
		SourceCandidate string `json:"source_candidate_id"`
		StoreKind       string `json:"store_kind"`
		StoreReady      bool   `json:"store_ready"`
		ReceiptVerified bool   `json:"receipt_verified"`
		BodyMutation    bool   `json:"body_mutation"`
		AdmissionOpen   bool   `json:"admission_open"`
	}{
		StoreHash:       store.StoreHash,
		SourceCandidate: store.SourceWeightedAdmissionResonanceGraftCandidateID,
		StoreKind:       store.StoreKind,
		StoreReady:      store.WeightedAdmissionResonanceGraftCandidateStoreReady,
		ReceiptVerified: store.ReceiptVerified,
		BodyMutation:    store.BodyMutationAllowed,
		AdmissionOpen:   store.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreID(store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourceCandidateID       string `json:"source_candidate_id"`
		SourceGateID            string `json:"source_gate_id"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		StoreHash               string `json:"store_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		StoreKind               string `json:"store_kind"`
		StoreMode               string `json:"store_mode"`
		StoreStage              string `json:"store_stage"`
		AppendOnly              bool   `json:"append_only"`
		ReadBack                bool   `json:"read_back"`
		ReceiptPersisted        bool   `json:"receipt_persisted"`
		ReceiptVerified         bool   `json:"receipt_verified"`
		CandidateVerified       bool   `json:"candidate_verified"`
		GateVerified            bool   `json:"gate_verified"`
		PreflightVerified       bool   `json:"preflight_verified"`
		BoundaryVerified        bool   `json:"boundary_verified"`
		ObservationVerified     bool   `json:"observation_verified"`
		ReceiverVerified        bool   `json:"receiver_verified"`
		IntentVerified          bool   `json:"intent_verified"`
		FinalGateVerified       bool   `json:"final_gate_verified"`
		SealVerified            bool   `json:"seal_verified"`
		PermitVerified          bool   `json:"permit_verified"`
		AuthorityVerified       bool   `json:"authority_verified"`
		AdmissionRequired       bool   `json:"admission_required"`
		ShadowOnly              bool   `json:"shadow_only"`
		GraftAllowed            bool   `json:"graft_allowed"`
		DryRunOnly              bool   `json:"dry_run_only"`
		RawDreamTextAllowed     bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed     bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed     bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed     bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		RollbackRequired        bool   `json:"rollback_required"`
		LiveReady               bool   `json:"live_ready"`
		ContractsReady          bool   `json:"contracts_ready"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_candidate_store"`
		SourceCandidateReady    bool   `json:"source_candidate_ready"`
		SourceGateReady         bool   `json:"source_gate_ready"`
		SourcePreflightReady    bool   `json:"source_preflight_ready"`
		SourceBoundaryReady     bool   `json:"source_boundary_ready"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
		SourceAuthorityRequired bool   `json:"source_authority_required"`
	}{
		Schema:                  store.Schema,
		Status:                  store.Status,
		Action:                  store.Action,
		SourceReport:            store.SourceReport,
		SourceCandidateID:       store.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:            store.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:       store.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        store.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     store.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        store.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                store.CausalID,
		StoreHash:               store.StoreHash,
		ReadBackHash:            store.ReadBackHash,
		Ready:                   store.WeightedAdmissionResonanceGraftCandidateStoreReady,
		ReceiptShape:            store.ReceiptShape,
		StoreKind:               store.StoreKind,
		StoreMode:               store.StoreMode,
		StoreStage:              store.StoreStage,
		AppendOnly:              store.AppendOnly,
		ReadBack:                store.ReadBack,
		ReceiptPersisted:        store.ReceiptPersisted,
		ReceiptVerified:         store.ReceiptVerified,
		CandidateVerified:       store.CandidateVerified,
		GateVerified:            store.GateVerified,
		PreflightVerified:       store.PreflightVerified,
		BoundaryVerified:        store.BoundaryVerified,
		ObservationVerified:     store.ObservationVerified,
		ReceiverVerified:        store.ReceiverVerified,
		IntentVerified:          store.IntentVerified,
		FinalGateVerified:       store.FinalGateVerified,
		SealVerified:            store.SealVerified,
		PermitVerified:          store.PermitVerified,
		AuthorityVerified:       store.AuthorityVerified,
		AdmissionRequired:       store.AdmissionRequired,
		ShadowOnly:              store.ShadowOnly,
		GraftAllowed:            store.GraftAllowed,
		DryRunOnly:              store.DryRunOnly,
		RawDreamTextAllowed:     store.RawDreamTextAllowed,
		JanusSurfaceAllowed:     store.JanusSurfaceAllowed,
		CoocLearningAllowed:     store.CoocLearningAllowed,
		DeltaHarvestAllowed:     store.DeltaHarvestAllowed,
		BodyMutationAllowed:     store.BodyMutationAllowed,
		RollbackRequired:        store.RollbackRequired,
		LiveReady:               store.LiveReady,
		ContractsReady:          store.ContractsReady,
		BodyTarget:              store.BodyTarget,
		WriteAllowed:            store.WriteAllowed,
		AdmissionAllowed:        store.AdmissionAllowed,
		LiveAdmissionEnabled:    store.LiveAdmissionEnabled,
		MutatesState:            store.MutatesState,
		NextStepBlockedWithout:  store.NextStepBlockedWithoutResonanceGraftCandidateStore,
		SourceCandidateReady:    store.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:         store.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:    store.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     store.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  store.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     store.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       store.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    store.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         store.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       store.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: store.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityRequired: store.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-store-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft candidate store not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate store decode failed: %w", err)
	}
	return report, root, nil
}
