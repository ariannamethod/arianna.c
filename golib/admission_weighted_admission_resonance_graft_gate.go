package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema = "arianna.live_route_weighted_admission_resonance_graft_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftGateReport struct {
	Schema                                               string `json:"schema"`
	Status                                               string `json:"status"`
	Target                                               string `json:"target"`
	TargetKind                                           string `json:"target_kind"`
	TargetMode                                           string `json:"target_mode"`
	Action                                               string `json:"action"`
	WeightedAdmissionResonanceGraftGateReady             bool   `json:"weighted_admission_resonance_graft_gate_ready"`
	WeightedAdmissionResonanceGraftPreflightConsumed     bool   `json:"weighted_admission_resonance_graft_preflight_consumed"`
	WeightedAdmissionResonanceGraftPreflightRequired     bool   `json:"weighted_admission_resonance_graft_preflight_required"`
	NextStepBlockedWithoutResonanceGraftGate             bool   `json:"next_step_blocked_without_resonance_graft_gate"`
	WeightedAdmissionResonanceGraftGateID                string `json:"weighted_admission_resonance_graft_gate_id"`
	ReceiptShape                                         string `json:"receipt_shape"`
	GateKind                                             string `json:"gate_kind"`
	GateMode                                             string `json:"gate_mode"`
	GateStage                                            string `json:"gate_stage"`
	CausalID                                             string `json:"causal_id"`
	GateHash                                             string `json:"gate_hash"`
	ReadBackHash                                         string `json:"read_back_hash"`
	PreflightVerified                                    bool   `json:"preflight_verified"`
	BoundaryVerified                                     bool   `json:"boundary_verified"`
	ObservationVerified                                  bool   `json:"observation_verified"`
	ReceiverVerified                                     bool   `json:"receiver_verified"`
	IntentVerified                                       bool   `json:"intent_verified"`
	FinalGateVerified                                    bool   `json:"final_gate_verified"`
	SealVerified                                         bool   `json:"seal_verified"`
	PermitVerified                                       bool   `json:"permit_verified"`
	AuthorityVerified                                    bool   `json:"authority_verified"`
	AdmissionRequired                                    bool   `json:"admission_required"`
	ShadowOnly                                           bool   `json:"shadow_only"`
	GraftAllowed                                         bool   `json:"graft_allowed"`
	DryRunOnly                                           bool   `json:"dry_run_only"`
	LiveReady                                            bool   `json:"live_ready"`
	RawDreamTextAllowed                                  bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                 bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                  bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                  bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                  bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                  bool   `json:"body_mutation_allowed"`
	RollbackRequired                                     bool   `json:"rollback_required"`
	SourceSchema                                         string `json:"source_schema"`
	SourceStatus                                         string `json:"source_status"`
	SourceTarget                                         string `json:"source_target"`
	SourceReport                                         string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftPreflightID     string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady  bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightCausal string `json:"source_weighted_admission_resonance_graft_preflight_causal_id"`
	SourceWeightedAdmissionResonanceGraftPreflightHash   string `json:"source_weighted_admission_resonance_graft_preflight_hash"`
	SourceWeightedAdmissionResonanceGraftPreflightRead   string `json:"source_weighted_admission_resonance_graft_preflight_read_back_hash"`
	SourcePreflightAction                                string `json:"source_preflight_action"`
	SourcePreflightReceiptShape                          string `json:"source_preflight_receipt_shape"`
	SourcePreflightKind                                  string `json:"source_preflight_kind"`
	SourcePreflightMode                                  string `json:"source_preflight_mode"`
	SourcePreflightStage                                 string `json:"source_preflight_stage"`
	SourcePreflightShadowOnly                            bool   `json:"source_preflight_shadow_only"`
	SourcePreflightGraftAllowed                          bool   `json:"source_preflight_graft_allowed"`
	SourcePreflightDryRunOnly                            bool   `json:"source_preflight_dry_run_only"`
	SourcePreflightLiveReady                             bool   `json:"source_preflight_live_ready"`
	SourcePreflightRawDreamTextAllowed                   bool   `json:"source_preflight_raw_dream_text_allowed"`
	SourcePreflightRawDreamTextObserved                  bool   `json:"source_preflight_raw_dream_text_observed"`
	SourcePreflightRawDreamTextForwarded                 bool   `json:"source_preflight_raw_dream_text_forwarded"`
	SourcePreflightJanusSurfaceAllowed                   bool   `json:"source_preflight_janus_surface_allowed"`
	SourcePreflightCoocLearningAllowed                   bool   `json:"source_preflight_cooc_learning_allowed"`
	SourcePreflightDeltaHarvestAllowed                   bool   `json:"source_preflight_delta_harvest_allowed"`
	SourcePreflightBodyMutationAllowed                   bool   `json:"source_preflight_body_mutation_allowed"`
	SourcePreflightRollbackRequired                      bool   `json:"source_preflight_rollback_required"`
	SourceGraftBoundarySchema                            string `json:"source_graft_boundary_schema"`
	SourceGraftBoundaryStatus                            string `json:"source_graft_boundary_status"`
	SourceGraftBoundaryTarget                            string `json:"source_graft_boundary_target"`
	SourceGraftBoundaryReport                            string `json:"source_graft_boundary_report"`
	SourceWeightedAdmissionResonanceGraftBoundaryID      string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady   bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryCausal  string `json:"source_weighted_admission_resonance_graft_boundary_causal_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryHash    string `json:"source_weighted_admission_resonance_graft_boundary_hash"`
	SourceWeightedAdmissionResonanceGraftBoundaryRead    string `json:"source_weighted_admission_resonance_graft_boundary_read_back_hash"`
	SourceBoundaryAction                                 string `json:"source_boundary_action"`
	SourceBoundaryReceiptShape                           string `json:"source_boundary_receipt_shape"`
	SourceBoundaryKind                                   string `json:"source_boundary_kind"`
	SourceBoundaryMode                                   string `json:"source_boundary_mode"`
	SourceBoundaryStage                                  string `json:"source_boundary_stage"`
	SourceBoundaryShadowOnly                             bool   `json:"source_boundary_shadow_only"`
	SourceBoundaryGraftAllowed                           bool   `json:"source_boundary_graft_allowed"`
	SourceBoundaryDryRunOnly                             bool   `json:"source_boundary_dry_run_only"`
	SourceBoundaryLiveReady                              bool   `json:"source_boundary_live_ready"`
	SourceBoundaryRawDreamTextAllowed                    bool   `json:"source_boundary_raw_dream_text_allowed"`
	SourceBoundaryRawDreamTextObserved                   bool   `json:"source_boundary_raw_dream_text_observed"`
	SourceBoundaryRawDreamTextForwarded                  bool   `json:"source_boundary_raw_dream_text_forwarded"`
	SourceBoundaryJanusSurfaceAllowed                    bool   `json:"source_boundary_janus_surface_allowed"`
	SourceBoundaryCoocLearningAllowed                    bool   `json:"source_boundary_cooc_learning_allowed"`
	SourceBoundaryDeltaHarvestAllowed                    bool   `json:"source_boundary_delta_harvest_allowed"`
	SourceBoundaryBodyMutationAllowed                    bool   `json:"source_boundary_body_mutation_allowed"`
	SourceBoundaryRollbackRequired                       bool   `json:"source_boundary_rollback_required"`
	SourceObservationSchema                              string `json:"source_observation_schema"`
	SourceObservationStatus                              string `json:"source_observation_status"`
	SourceObservationTarget                              string `json:"source_observation_target"`
	SourceObservationReport                              string `json:"source_observation_report"`
	SourceWeightedAdmissionResonanceObservationID        string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady     bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceObservationCausal    string `json:"source_weighted_admission_resonance_observation_causal_id"`
	SourceWeightedAdmissionResonanceObservationAppend    string `json:"source_weighted_admission_resonance_observation_append_hash"`
	SourceWeightedAdmissionResonanceObservationRead      string `json:"source_weighted_admission_resonance_observation_read_back_hash"`
	SourceObserver                                       string `json:"source_observer"`
	SourceObserverKind                                   string `json:"source_observer_kind"`
	SourceObservationKind                                string `json:"source_observation_kind"`
	SourceObservationMode                                string `json:"source_observation_mode"`
	SourceAppendOnly                                     bool   `json:"source_append_only"`
	SourceReadBack                                       bool   `json:"source_read_back"`
	SourceReceiptVerified                                bool   `json:"source_receipt_verified"`
	SourceDryRunOnly                                     bool   `json:"source_dry_run_only"`
	SourceObservationRawDreamTextObserved                bool   `json:"source_observation_raw_dream_text_observed"`
	SourceObservationRawDreamTextForwarded               bool   `json:"source_observation_raw_dream_text_forwarded"`
	SourceObservationJanusSurfaceAllowed                 bool   `json:"source_observation_janus_surface_allowed"`
	SourceObservationCoocLearningAllowed                 bool   `json:"source_observation_cooc_learning_allowed"`
	SourceObservationDeltaHarvestAllowed                 bool   `json:"source_observation_delta_harvest_allowed"`
	SourceObservationBodyMutationAllowed                 bool   `json:"source_observation_body_mutation_allowed"`
	SourceObservationRollbackRequired                    bool   `json:"source_observation_rollback_required"`
	SourceResonanceReceiverReport                        string `json:"source_resonance_receiver_report"`
	SourceResonanceIntentReport                          string `json:"source_resonance_intent_report"`
	SourceFinalGateReport                                string `json:"source_final_gate_report"`
	SourceSealReport                                     string `json:"source_seal_report"`
	SourcePermitReport                                   string `json:"source_permit_report"`
	SourceAuthorityReport                                string `json:"source_authority_report"`
	SourceContractReport                                 string `json:"source_contract_report"`
	SourcePreconditionReport                             string `json:"source_precondition_report"`
	SourceReadinessReport                                string `json:"source_readiness_report"`
	SourceBodyWorkdir                                    string `json:"source_body_workdir"`
	SourceBoundaryReport                                 string `json:"source_boundary_report"`
	SourceProofLog                                       string `json:"source_proof_log"`
	SourceFinalGateLog                                   string `json:"source_final_gate_log"`
	SourceWeightedAdmissionResonanceReceiverID           string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady        bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceReceiverCausal       string `json:"source_weighted_admission_resonance_receiver_causal_id"`
	SourceReceiverPreStateHash                           string `json:"source_receiver_pre_state_hash"`
	SourceReceiverPostStateHash                          string `json:"source_receiver_post_state_hash"`
	SourceReceiverStateDeltaHash                         string `json:"source_receiver_state_delta_hash"`
	SourceWeightedAdmissionResonanceIntentConsumed       bool   `json:"source_weighted_admission_resonance_intent_consumed"`
	SourceWeightedAdmissionResonanceIntentRequired       bool   `json:"source_weighted_admission_resonance_intent_required"`
	SourceWeightedAdmissionResonanceIntentReady          bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateConsumed             bool   `json:"source_weighted_admission_final_gate_consumed"`
	SourceWeightedAdmissionFinalGateRequired             bool   `json:"source_weighted_admission_final_gate_required"`
	SourceWeightedAdmissionFinalGateReady                bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed                  bool   `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired                  bool   `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady                     bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed                bool   `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired                bool   `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady                   bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed             bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired             bool   `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested                          bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                               bool   `json:"source_permit_key_matched"`
	SourceRawDreamTextAllowed                            bool   `json:"source_raw_dream_text_allowed"`
	SourceRawDreamTextObserved                           bool   `json:"source_raw_dream_text_observed"`
	SourceRawDreamTextForwarded                          bool   `json:"source_raw_dream_text_forwarded"`
	SourceJanusSurfaceAllowed                            bool   `json:"source_janus_surface_allowed"`
	SourceCoocLearningAllowed                            bool   `json:"source_cooc_learning_allowed"`
	SourceDeltaHarvestAllowed                            bool   `json:"source_delta_harvest_allowed"`
	SourceBodyMutationAllowed                            bool   `json:"source_body_mutation_allowed"`
	SourceRollbackRequired                               bool   `json:"source_rollback_required"`
	SourcePreStateHashRequired                           bool   `json:"source_pre_state_hash_required"`
	SourcePostStateHashRequired                          bool   `json:"source_post_state_hash_required"`
	BodySmokeWeighted                                    bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                     bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                  bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                         bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                              bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                               bool   `json:"source_authority_granted"`
	AuthorityGranted                                     bool   `json:"authority_granted"`
	ContractsReady                                       bool   `json:"contracts_ready"`
	WriteAllowed                                         bool   `json:"write_allowed"`
	AdmissionAllowed                                     bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                 bool   `json:"live_admission_enabled"`
	MutatesState                                         bool   `json:"mutates_state"`
	BodyTarget                                           string `json:"body_target"`
	Passed                                               bool   `json:"passed"`
	Reason                                               string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-gate RESONANCE_GRAFT_PREFLIGHT_REPORT RESONANCE_GRAFT_GATE_REPORT")
	}
	preflightPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft gate output path missing")
	}
	preflight, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportForAssert(preflightPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportError(preflight, root); err != nil {
		return err
	}
	gate := admissionLiveRouteWeightedAdmissionResonanceGraftGateReport{
		Schema:                                   admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema,
		Status:                                   "shadow_graft_gate_ready_dry_run",
		Target:                                   "resonance",
		TargetKind:                               "weighted_internal_world_shadow_graft_gate",
		TargetMode:                               "receipt_only_closed_gate_dry_run",
		Action:                                   "gate_weighted_resonance_shadow_graft_dry_run",
		WeightedAdmissionResonanceGraftGateReady: true,
		WeightedAdmissionResonanceGraftPreflightConsumed: true,
		WeightedAdmissionResonanceGraftPreflightRequired: true,
		NextStepBlockedWithoutResonanceGraftGate:         true,
		ReceiptShape:                                     "weighted_resonance_shadow_graft_gate_contract",
		GateKind:                                         "shadow_graft_gate",
		GateMode:                                         "no_mutation_gate",
		GateStage:                                        "pre_live_graft_gate",
		PreflightVerified:                                true,
		BoundaryVerified:                                 preflight.BoundaryVerified,
		ObservationVerified:                              preflight.ObservationVerified,
		ReceiverVerified:                                 preflight.ReceiverVerified,
		IntentVerified:                                   preflight.IntentVerified,
		FinalGateVerified:                                preflight.FinalGateVerified,
		SealVerified:                                     preflight.SealVerified,
		PermitVerified:                                   preflight.PermitVerified,
		AuthorityVerified:                                preflight.AuthorityVerified,
		AdmissionRequired:                                true,
		ShadowOnly:                                       true,
		GraftAllowed:                                     false,
		DryRunOnly:                                       true,
		LiveReady:                                        true,
		RawDreamTextAllowed:                              false,
		RawDreamTextObserved:                             false,
		RawDreamTextForwarded:                            false,
		JanusSurfaceAllowed:                              false,
		CoocLearningAllowed:                              false,
		DeltaHarvestAllowed:                              false,
		BodyMutationAllowed:                              false,
		RollbackRequired:                                 true,
		SourceSchema:                                     preflight.Schema,
		SourceStatus:                                     preflight.Status,
		SourceTarget:                                     preflight.Target,
		SourceReport:                                     preflightPath,
		SourceWeightedAdmissionResonanceGraftPreflightID:     preflight.WeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:  preflight.WeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftPreflightCausal: preflight.CausalID,
		SourceWeightedAdmissionResonanceGraftPreflightHash:   preflight.PreflightHash,
		SourceWeightedAdmissionResonanceGraftPreflightRead:   preflight.ReadBackHash,
		SourcePreflightAction:                                preflight.Action,
		SourcePreflightReceiptShape:                          preflight.ReceiptShape,
		SourcePreflightKind:                                  preflight.PreflightKind,
		SourcePreflightMode:                                  preflight.PreflightMode,
		SourcePreflightStage:                                 preflight.PreflightStage,
		SourcePreflightShadowOnly:                            preflight.ShadowOnly,
		SourcePreflightGraftAllowed:                          preflight.GraftAllowed,
		SourcePreflightDryRunOnly:                            preflight.DryRunOnly,
		SourcePreflightLiveReady:                             preflight.LiveReady,
		SourcePreflightRawDreamTextAllowed:                   preflight.RawDreamTextAllowed,
		SourcePreflightRawDreamTextObserved:                  preflight.RawDreamTextObserved,
		SourcePreflightRawDreamTextForwarded:                 preflight.RawDreamTextForwarded,
		SourcePreflightJanusSurfaceAllowed:                   preflight.JanusSurfaceAllowed,
		SourcePreflightCoocLearningAllowed:                   preflight.CoocLearningAllowed,
		SourcePreflightDeltaHarvestAllowed:                   preflight.DeltaHarvestAllowed,
		SourcePreflightBodyMutationAllowed:                   preflight.BodyMutationAllowed,
		SourcePreflightRollbackRequired:                      preflight.RollbackRequired,
		SourceGraftBoundarySchema:                            preflight.SourceSchema,
		SourceGraftBoundaryStatus:                            preflight.SourceStatus,
		SourceGraftBoundaryTarget:                            preflight.SourceTarget,
		SourceGraftBoundaryReport:                            preflight.SourceReport,
		SourceWeightedAdmissionResonanceGraftBoundaryID:      preflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:   preflight.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceGraftBoundaryCausal:  preflight.SourceWeightedAdmissionResonanceGraftBoundaryCausal,
		SourceWeightedAdmissionResonanceGraftBoundaryHash:    preflight.SourceWeightedAdmissionResonanceGraftBoundaryHash,
		SourceWeightedAdmissionResonanceGraftBoundaryRead:    preflight.SourceWeightedAdmissionResonanceGraftBoundaryRead,
		SourceBoundaryAction:                                 preflight.SourceBoundaryAction,
		SourceBoundaryReceiptShape:                           preflight.SourceBoundaryReceiptShape,
		SourceBoundaryKind:                                   preflight.SourceBoundaryKind,
		SourceBoundaryMode:                                   preflight.SourceBoundaryMode,
		SourceBoundaryStage:                                  preflight.SourceBoundaryStage,
		SourceBoundaryShadowOnly:                             preflight.SourceBoundaryShadowOnly,
		SourceBoundaryGraftAllowed:                           preflight.SourceBoundaryGraftAllowed,
		SourceBoundaryDryRunOnly:                             preflight.SourceBoundaryDryRunOnly,
		SourceBoundaryLiveReady:                              preflight.SourceBoundaryLiveReady,
		SourceBoundaryRawDreamTextAllowed:                    preflight.SourceBoundaryRawDreamTextAllowed,
		SourceBoundaryRawDreamTextObserved:                   preflight.SourceBoundaryRawDreamTextObserved,
		SourceBoundaryRawDreamTextForwarded:                  preflight.SourceBoundaryRawDreamTextForwarded,
		SourceBoundaryJanusSurfaceAllowed:                    preflight.SourceBoundaryJanusSurfaceAllowed,
		SourceBoundaryCoocLearningAllowed:                    preflight.SourceBoundaryCoocLearningAllowed,
		SourceBoundaryDeltaHarvestAllowed:                    preflight.SourceBoundaryDeltaHarvestAllowed,
		SourceBoundaryBodyMutationAllowed:                    preflight.SourceBoundaryBodyMutationAllowed,
		SourceBoundaryRollbackRequired:                       preflight.SourceBoundaryRollbackRequired,
		SourceObservationSchema:                              preflight.SourceObservationSchema,
		SourceObservationStatus:                              preflight.SourceObservationStatus,
		SourceObservationTarget:                              preflight.SourceObservationTarget,
		SourceObservationReport:                              preflight.SourceObservationReport,
		SourceWeightedAdmissionResonanceObservationID:        preflight.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:     preflight.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceObservationCausal:    preflight.SourceWeightedAdmissionResonanceObservationCausal,
		SourceWeightedAdmissionResonanceObservationAppend:    preflight.SourceWeightedAdmissionResonanceObservationAppend,
		SourceWeightedAdmissionResonanceObservationRead:      preflight.SourceWeightedAdmissionResonanceObservationRead,
		SourceObserver:                                       preflight.SourceObserver,
		SourceObserverKind:                                   preflight.SourceObserverKind,
		SourceObservationKind:                                preflight.SourceObservationKind,
		SourceObservationMode:                                preflight.SourceObservationMode,
		SourceAppendOnly:                                     preflight.SourceAppendOnly,
		SourceReadBack:                                       preflight.SourceReadBack,
		SourceReceiptVerified:                                preflight.SourceReceiptVerified,
		SourceDryRunOnly:                                     preflight.SourceDryRunOnly,
		SourceObservationRawDreamTextObserved:                preflight.SourceObservationRawDreamTextObserved,
		SourceObservationRawDreamTextForwarded:               preflight.SourceObservationRawDreamTextForwarded,
		SourceObservationJanusSurfaceAllowed:                 preflight.SourceObservationJanusSurfaceAllowed,
		SourceObservationCoocLearningAllowed:                 preflight.SourceObservationCoocLearningAllowed,
		SourceObservationDeltaHarvestAllowed:                 preflight.SourceObservationDeltaHarvestAllowed,
		SourceObservationBodyMutationAllowed:                 preflight.SourceObservationBodyMutationAllowed,
		SourceObservationRollbackRequired:                    preflight.SourceObservationRollbackRequired,
		SourceResonanceReceiverReport:                        preflight.SourceResonanceReceiverReport,
		SourceResonanceIntentReport:                          preflight.SourceResonanceIntentReport,
		SourceFinalGateReport:                                preflight.SourceFinalGateReport,
		SourceSealReport:                                     preflight.SourceSealReport,
		SourcePermitReport:                                   preflight.SourcePermitReport,
		SourceAuthorityReport:                                preflight.SourceAuthorityReport,
		SourceContractReport:                                 preflight.SourceContractReport,
		SourcePreconditionReport:                             preflight.SourcePreconditionReport,
		SourceReadinessReport:                                preflight.SourceReadinessReport,
		SourceBodyWorkdir:                                    preflight.SourceBodyWorkdir,
		SourceBoundaryReport:                                 preflight.SourceBoundaryReport,
		SourceProofLog:                                       preflight.SourceProofLog,
		SourceFinalGateLog:                                   preflight.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceReceiverID:           preflight.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:        preflight.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceReceiverCausal:       preflight.SourceWeightedAdmissionResonanceReceiverCausal,
		SourceReceiverPreStateHash:                           preflight.SourceReceiverPreStateHash,
		SourceReceiverPostStateHash:                          preflight.SourceReceiverPostStateHash,
		SourceReceiverStateDeltaHash:                         preflight.SourceReceiverStateDeltaHash,
		SourceWeightedAdmissionResonanceIntentConsumed:       preflight.SourceWeightedAdmissionResonanceIntentConsumed,
		SourceWeightedAdmissionResonanceIntentRequired:       preflight.SourceWeightedAdmissionResonanceIntentRequired,
		SourceWeightedAdmissionResonanceIntentReady:          preflight.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:             preflight.SourceWeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:             preflight.SourceWeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:                preflight.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:                  preflight.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:                  preflight.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:                     preflight.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:                preflight.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:                preflight.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:                   preflight.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:             preflight.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:             preflight.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                          preflight.SourceManualPermitRequested,
		SourcePermitKeyMatched:                               preflight.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                            preflight.SourceRawDreamTextAllowed,
		SourceRawDreamTextObserved:                           preflight.SourceRawDreamTextObserved,
		SourceRawDreamTextForwarded:                          preflight.SourceRawDreamTextForwarded,
		SourceJanusSurfaceAllowed:                            preflight.SourceJanusSurfaceAllowed,
		SourceCoocLearningAllowed:                            preflight.SourceCoocLearningAllowed,
		SourceDeltaHarvestAllowed:                            preflight.SourceDeltaHarvestAllowed,
		SourceBodyMutationAllowed:                            preflight.SourceBodyMutationAllowed,
		SourceRollbackRequired:                               preflight.SourceRollbackRequired,
		SourcePreStateHashRequired:                           preflight.SourcePreStateHashRequired,
		SourcePostStateHashRequired:                          preflight.SourcePostStateHashRequired,
		BodySmokeWeighted:                                    preflight.BodySmokeWeighted,
		NanoDirectRunner:                                     preflight.NanoDirectRunner,
		NanoDirectFinalGate:                                  preflight.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                         preflight.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                              preflight.BoundaryReportFullChain,
		SourceAuthorityGranted:                               preflight.SourceAuthorityGranted,
		AuthorityGranted:                                     false,
		ContractsReady:                                       false,
		WriteAllowed:                                         false,
		AdmissionAllowed:                                     false,
		LiveAdmissionEnabled:                                 false,
		MutatesState:                                         false,
		BodyTarget:                                           "none",
		Passed:                                               true,
		Reason:                                               "weighted resonance shadow graft gate prepared without body mutation",
	}
	gate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftGateCausalID(gate)
	gate.GateHash = admissionLiveRouteWeightedAdmissionResonanceGraftGateHash(gate)
	gate.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftGateReadBackHash(gate)
	gate.WeightedAdmissionResonanceGraftGateID = admissionLiveRouteWeightedAdmissionResonanceGraftGateID(gate)
	if gate.CausalID == "" ||
		gate.GateHash == "" ||
		gate.ReadBackHash == "" ||
		gate.WeightedAdmissionResonanceGraftGateID == "" ||
		gate.GateHash == gate.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(gate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-gate] pass: resonance_graft_gate_report=%s resonance_graft_preflight_report=%s\n", outputPath, preflightPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema {
		return fmt.Errorf("weighted admission resonance graft gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema)
	}
	if report.Status != "shadow_graft_gate_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate status mismatch: got %q want %q", report.Status, "shadow_graft_gate_ready_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft gate target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_gate" {
		return fmt.Errorf("weighted admission resonance graft gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_gate")
	}
	if report.TargetMode != "receipt_only_closed_gate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_gate_dry_run")
	}
	if report.Action != "gate_weighted_resonance_shadow_graft_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate action mismatch: got %q want %q", report.Action, "gate_weighted_resonance_shadow_graft_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_gate_contract" {
		return fmt.Errorf("weighted admission resonance graft gate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_gate_contract")
	}
	if report.GateKind != "shadow_graft_gate" {
		return fmt.Errorf("weighted admission resonance graft gate gate_kind mismatch: got %q want %q", report.GateKind, "shadow_graft_gate")
	}
	if report.GateMode != "no_mutation_gate" {
		return fmt.Errorf("weighted admission resonance graft gate gate_mode mismatch: got %q want %q", report.GateMode, "no_mutation_gate")
	}
	if report.GateStage != "pre_live_graft_gate" {
		return fmt.Errorf("weighted admission resonance graft gate gate_stage mismatch: got %q want %q", report.GateStage, "pre_live_graft_gate")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_gate_ready", report.WeightedAdmissionResonanceGraftGateReady},
		{"weighted_admission_resonance_graft_preflight_consumed", report.WeightedAdmissionResonanceGraftPreflightConsumed},
		{"weighted_admission_resonance_graft_preflight_required", report.WeightedAdmissionResonanceGraftPreflightRequired},
		{"next_step_blocked_without_resonance_graft_gate", report.NextStepBlockedWithoutResonanceGraftGate},
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
		{"source_weighted_admission_resonance_graft_preflight_ready", report.SourceWeightedAdmissionResonanceGraftPreflightReady},
		{"source_preflight_shadow_only", report.SourcePreflightShadowOnly},
		{"source_preflight_dry_run_only", report.SourcePreflightDryRunOnly},
		{"source_preflight_live_ready", report.SourcePreflightLiveReady},
		{"source_preflight_rollback_required", report.SourcePreflightRollbackRequired},
		{"source_weighted_admission_resonance_graft_boundary_ready", report.SourceWeightedAdmissionResonanceGraftBoundaryReady},
		{"source_boundary_shadow_only", report.SourceBoundaryShadowOnly},
		{"source_boundary_dry_run_only", report.SourceBoundaryDryRunOnly},
		{"source_boundary_live_ready", report.SourceBoundaryLiveReady},
		{"source_boundary_rollback_required", report.SourceBoundaryRollbackRequired},
		{"source_weighted_admission_resonance_observation_ready", report.SourceWeightedAdmissionResonanceObservationReady},
		{"source_append_only", report.SourceAppendOnly},
		{"source_read_back", report.SourceReadBack},
		{"source_receipt_verified", report.SourceReceiptVerified},
		{"source_dry_run_only", report.SourceDryRunOnly},
		{"source_observation_rollback_required", report.SourceObservationRollbackRequired},
		{"source_weighted_admission_resonance_receiver_ready", report.SourceWeightedAdmissionResonanceReceiverReady},
		{"source_weighted_admission_resonance_intent_consumed", report.SourceWeightedAdmissionResonanceIntentConsumed},
		{"source_weighted_admission_resonance_intent_required", report.SourceWeightedAdmissionResonanceIntentRequired},
		{"source_weighted_admission_resonance_intent_ready", report.SourceWeightedAdmissionResonanceIntentReady},
		{"source_weighted_admission_final_gate_consumed", report.SourceWeightedAdmissionFinalGateConsumed},
		{"source_weighted_admission_final_gate_required", report.SourceWeightedAdmissionFinalGateRequired},
		{"source_weighted_admission_final_gate_ready", report.SourceWeightedAdmissionFinalGateReady},
		{"source_weighted_admission_seal_consumed", report.SourceWeightedAdmissionSealConsumed},
		{"source_weighted_admission_seal_required", report.SourceWeightedAdmissionSealRequired},
		{"source_weighted_admission_seal_ready", report.SourceWeightedAdmissionSealReady},
		{"source_weighted_admission_permit_consumed", report.SourceWeightedAdmissionPermitConsumed},
		{"source_weighted_admission_permit_required", report.SourceWeightedAdmissionPermitRequired},
		{"source_weighted_admission_permit_ready", report.SourceWeightedAdmissionPermitReady},
		{"source_weighted_admission_authority_consumed", report.SourceWeightedAdmissionAuthorityConsumed},
		{"source_weighted_admission_authority_required", report.SourceWeightedAdmissionAuthorityRequired},
		{"source_manual_permit_requested", report.SourceManualPermitRequested},
		{"source_permit_key_matched", report.SourcePermitKeyMatched},
		{"source_rollback_required", report.SourceRollbackRequired},
		{"source_pre_state_hash_required", report.SourcePreStateHashRequired},
		{"source_post_state_hash_required", report.SourcePostStateHashRequired},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft gate %s not ready", required.name)
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
		{"source_preflight_graft_allowed", report.SourcePreflightGraftAllowed},
		{"source_preflight_raw_dream_text_allowed", report.SourcePreflightRawDreamTextAllowed},
		{"source_preflight_raw_dream_text_observed", report.SourcePreflightRawDreamTextObserved},
		{"source_preflight_raw_dream_text_forwarded", report.SourcePreflightRawDreamTextForwarded},
		{"source_preflight_janus_surface_allowed", report.SourcePreflightJanusSurfaceAllowed},
		{"source_preflight_cooc_learning_allowed", report.SourcePreflightCoocLearningAllowed},
		{"source_preflight_delta_harvest_allowed", report.SourcePreflightDeltaHarvestAllowed},
		{"source_preflight_body_mutation_allowed", report.SourcePreflightBodyMutationAllowed},
		{"source_boundary_graft_allowed", report.SourceBoundaryGraftAllowed},
		{"source_boundary_raw_dream_text_allowed", report.SourceBoundaryRawDreamTextAllowed},
		{"source_boundary_raw_dream_text_observed", report.SourceBoundaryRawDreamTextObserved},
		{"source_boundary_raw_dream_text_forwarded", report.SourceBoundaryRawDreamTextForwarded},
		{"source_boundary_janus_surface_allowed", report.SourceBoundaryJanusSurfaceAllowed},
		{"source_boundary_cooc_learning_allowed", report.SourceBoundaryCoocLearningAllowed},
		{"source_boundary_delta_harvest_allowed", report.SourceBoundaryDeltaHarvestAllowed},
		{"source_boundary_body_mutation_allowed", report.SourceBoundaryBodyMutationAllowed},
		{"source_observation_raw_dream_text_observed", report.SourceObservationRawDreamTextObserved},
		{"source_observation_raw_dream_text_forwarded", report.SourceObservationRawDreamTextForwarded},
		{"source_observation_janus_surface_allowed", report.SourceObservationJanusSurfaceAllowed},
		{"source_observation_cooc_learning_allowed", report.SourceObservationCoocLearningAllowed},
		{"source_observation_delta_harvest_allowed", report.SourceObservationDeltaHarvestAllowed},
		{"source_observation_body_mutation_allowed", report.SourceObservationBodyMutationAllowed},
		{"source_raw_dream_text_allowed", report.SourceRawDreamTextAllowed},
		{"source_raw_dream_text_observed", report.SourceRawDreamTextObserved},
		{"source_raw_dream_text_forwarded", report.SourceRawDreamTextForwarded},
		{"source_janus_surface_allowed", report.SourceJanusSurfaceAllowed},
		{"source_cooc_learning_allowed", report.SourceCoocLearningAllowed},
		{"source_delta_harvest_allowed", report.SourceDeltaHarvestAllowed},
		{"source_body_mutation_allowed", report.SourceBodyMutationAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft gate opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_preflight_id", report.SourceWeightedAdmissionResonanceGraftPreflightID},
		{"source_weighted_admission_resonance_graft_preflight_causal_id", report.SourceWeightedAdmissionResonanceGraftPreflightCausal},
		{"source_weighted_admission_resonance_graft_preflight_hash", report.SourceWeightedAdmissionResonanceGraftPreflightHash},
		{"source_weighted_admission_resonance_graft_preflight_read_back_hash", report.SourceWeightedAdmissionResonanceGraftPreflightRead},
		{"source_graft_boundary_report", report.SourceGraftBoundaryReport},
		{"source_weighted_admission_resonance_graft_boundary_id", report.SourceWeightedAdmissionResonanceGraftBoundaryID},
		{"source_weighted_admission_resonance_graft_boundary_causal_id", report.SourceWeightedAdmissionResonanceGraftBoundaryCausal},
		{"source_weighted_admission_resonance_graft_boundary_hash", report.SourceWeightedAdmissionResonanceGraftBoundaryHash},
		{"source_weighted_admission_resonance_graft_boundary_read_back_hash", report.SourceWeightedAdmissionResonanceGraftBoundaryRead},
		{"source_observation_report", report.SourceObservationReport},
		{"source_weighted_admission_resonance_observation_id", report.SourceWeightedAdmissionResonanceObservationID},
		{"source_weighted_admission_resonance_observation_causal_id", report.SourceWeightedAdmissionResonanceObservationCausal},
		{"source_weighted_admission_resonance_observation_append_hash", report.SourceWeightedAdmissionResonanceObservationAppend},
		{"source_weighted_admission_resonance_observation_read_back_hash", report.SourceWeightedAdmissionResonanceObservationRead},
		{"source_resonance_receiver_report", report.SourceResonanceReceiverReport},
		{"source_resonance_intent_report", report.SourceResonanceIntentReport},
		{"source_final_gate_report", report.SourceFinalGateReport},
		{"source_seal_report", report.SourceSealReport},
		{"source_permit_report", report.SourcePermitReport},
		{"source_authority_report", report.SourceAuthorityReport},
		{"source_contract_report", report.SourceContractReport},
		{"source_precondition_report", report.SourcePreconditionReport},
		{"source_readiness_report", report.SourceReadinessReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
		{"source_weighted_admission_resonance_receiver_causal_id", report.SourceWeightedAdmissionResonanceReceiverCausal},
		{"source_receiver_pre_state_hash", report.SourceReceiverPreStateHash},
		{"source_receiver_post_state_hash", report.SourceReceiverPostStateHash},
		{"source_receiver_state_delta_hash", report.SourceReceiverStateDeltaHash},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission resonance graft gate %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema)
	}
	if report.SourceStatus != "shadow_graft_preflight_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_preflight_ready_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft gate source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourcePreflightAction != "prepare_weighted_resonance_shadow_graft_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate source_preflight_action mismatch: got %q want %q", report.SourcePreflightAction, "prepare_weighted_resonance_shadow_graft_preflight_dry_run")
	}
	if report.SourcePreflightReceiptShape != "weighted_resonance_shadow_graft_preflight_contract" {
		return fmt.Errorf("weighted admission resonance graft gate source_preflight_receipt_shape mismatch: got %q want %q", report.SourcePreflightReceiptShape, "weighted_resonance_shadow_graft_preflight_contract")
	}
	if report.SourcePreflightKind != "shadow_graft_preflight" ||
		report.SourcePreflightMode != "no_mutation_preflight" ||
		report.SourcePreflightStage != "pre_live_graft_admission" {
		return fmt.Errorf("weighted admission resonance graft gate source preflight shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft gate source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightCausal, "weighted-resonance-graft-preflight-causal-") {
		return fmt.Errorf("weighted admission resonance graft gate source preflight causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightHash, "weighted-resonance-graft-preflight-") {
		return fmt.Errorf("weighted admission resonance graft gate source preflight hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightRead, "weighted-resonance-graft-preflight-read-") ||
		report.SourceWeightedAdmissionResonanceGraftPreflightHash == report.SourceWeightedAdmissionResonanceGraftPreflightRead {
		return fmt.Errorf("weighted admission resonance graft gate source preflight read-back mismatch")
	}
	if report.SourceGraftBoundarySchema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft gate source_graft_boundary_schema mismatch: got %q want %q", report.SourceGraftBoundarySchema, admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema)
	}
	if report.SourceGraftBoundaryStatus != "shadow_graft_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate source_graft_boundary_status mismatch: got %q want %q", report.SourceGraftBoundaryStatus, "shadow_graft_boundary_declared_dry_run")
	}
	if report.SourceGraftBoundaryTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft gate source_graft_boundary_target mismatch: got %q want %q", report.SourceGraftBoundaryTarget, "resonance")
	}
	if report.SourceBoundaryAction != "declare_weighted_resonance_shadow_graft_boundary_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate source_boundary_action mismatch: got %q want %q", report.SourceBoundaryAction, "declare_weighted_resonance_shadow_graft_boundary_dry_run")
	}
	if report.SourceBoundaryReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" {
		return fmt.Errorf("weighted admission resonance graft gate source_boundary_receipt_shape mismatch: got %q want %q", report.SourceBoundaryReceiptShape, "weighted_resonance_observation_shadow_graft_boundary")
	}
	if report.SourceBoundaryKind != "shadow_graft_boundary" ||
		report.SourceBoundaryMode != "no_mutation_receipt" ||
		report.SourceBoundaryStage != "pre_live_graft" {
		return fmt.Errorf("weighted admission resonance graft gate source boundary shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft gate source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryCausal, "weighted-resonance-graft-boundary-causal-") {
		return fmt.Errorf("weighted admission resonance graft gate source boundary causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryHash, "weighted-resonance-graft-boundary-") {
		return fmt.Errorf("weighted admission resonance graft gate source boundary hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryRead, "weighted-resonance-graft-boundary-read-") ||
		report.SourceWeightedAdmissionResonanceGraftBoundaryHash == report.SourceWeightedAdmissionResonanceGraftBoundaryRead {
		return fmt.Errorf("weighted admission resonance graft gate source boundary read-back mismatch")
	}
	if report.SourceObservationSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema {
		return fmt.Errorf("weighted admission resonance graft gate source_observation_schema mismatch: got %q want %q", report.SourceObservationSchema, admissionLiveRouteWeightedAdmissionResonanceObservationSchema)
	}
	if report.SourceObservationStatus != "observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft gate source_observation_status mismatch: got %q want %q", report.SourceObservationStatus, "observation_recorded_dry_run")
	}
	if report.SourceObservationTarget != "resonance" ||
		report.SourceObserver != "resonance" ||
		report.SourceObserverKind != "internal_world" ||
		report.SourceObservationKind != "weighted_receiver_state_proof" ||
		report.SourceObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft gate source observation shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft gate source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") {
		return fmt.Errorf("weighted admission resonance graft gate source observation causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") {
		return fmt.Errorf("weighted admission resonance graft gate source observation append prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") {
		return fmt.Errorf("weighted admission resonance graft gate source observation read-back prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft gate source receiver id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") {
		return fmt.Errorf("weighted admission resonance graft gate source receiver causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(report.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(report.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		report.SourceReceiverPreStateHash == report.SourceReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft gate source receiver state proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft gate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft gate causal_id mismatch")
	}
	if report.GateHash == "" || report.GateHash != admissionLiveRouteWeightedAdmissionResonanceGraftGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft gate gate_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft gate read_back_hash mismatch")
	}
	if report.GateHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft gate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftGateID == "" || report.WeightedAdmissionResonanceGraftGateID != admissionLiveRouteWeightedAdmissionResonanceGraftGateID(report) {
		return fmt.Errorf("weighted admission resonance graft gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft gate prepared without body mutation" {
		return fmt.Errorf("weighted admission resonance graft gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftGateCausalID(gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport) string {
	h := hashJSON(struct {
		SourcePreflightID           string `json:"source_preflight_id"`
		SourcePreflightReadBackHash string `json:"source_preflight_read_back_hash"`
		SourceBoundaryID            string `json:"source_boundary_id"`
		SourceObservationID         string `json:"source_observation_id"`
		SourceReceiverID            string `json:"source_receiver_id"`
		Target                      string `json:"target"`
		GateKind                    string `json:"gate_kind"`
		GateStage                   string `json:"gate_stage"`
	}{
		SourcePreflightID:           gate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourcePreflightReadBackHash: gate.SourceWeightedAdmissionResonanceGraftPreflightRead,
		SourceBoundaryID:            gate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:         gate.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:            gate.SourceWeightedAdmissionResonanceReceiverID,
		Target:                      gate.Target,
		GateKind:                    gate.GateKind,
		GateStage:                   gate.GateStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftGateHash(gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport) string {
	h := hashJSON(struct {
		CausalID                string `json:"causal_id"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourcePreflightHash     string `json:"source_preflight_hash"`
		SourcePreflightReadBack string `json:"source_preflight_read_back_hash"`
		GateMode                string `json:"gate_mode"`
		AdmissionRequired       bool   `json:"admission_required"`
		ShadowOnly              bool   `json:"shadow_only"`
		DryRunOnly              bool   `json:"dry_run_only"`
		GraftAllowed            bool   `json:"graft_allowed"`
	}{
		CausalID:                gate.CausalID,
		SourcePreflightID:       gate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourcePreflightHash:     gate.SourceWeightedAdmissionResonanceGraftPreflightHash,
		SourcePreflightReadBack: gate.SourceWeightedAdmissionResonanceGraftPreflightRead,
		GateMode:                gate.GateMode,
		AdmissionRequired:       gate.AdmissionRequired,
		ShadowOnly:              gate.ShadowOnly,
		DryRunOnly:              gate.DryRunOnly,
		GraftAllowed:            gate.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftGateReadBackHash(gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport) string {
	h := hashJSON(struct {
		GateHash        string `json:"gate_hash"`
		SourcePreflight string `json:"source_preflight_id"`
		GateKind        string `json:"gate_kind"`
		GateReady       bool   `json:"gate_ready"`
		BodyMutation    bool   `json:"body_mutation"`
		AdmissionOpen   bool   `json:"admission_open"`
	}{
		GateHash:        gate.GateHash,
		SourcePreflight: gate.SourceWeightedAdmissionResonanceGraftPreflightID,
		GateKind:        gate.GateKind,
		GateReady:       gate.WeightedAdmissionResonanceGraftGateReady,
		BodyMutation:    gate.BodyMutationAllowed,
		AdmissionOpen:   gate.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftGateID(gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		GateHash                string `json:"gate_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		GateKind                string `json:"gate_kind"`
		GateMode                string `json:"gate_mode"`
		GateStage               string `json:"gate_stage"`
		BoundaryVerified        bool   `json:"boundary_verified"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_gate"`
		SourcePreflightReady    bool   `json:"source_preflight_ready"`
		SourceBoundaryReady     bool   `json:"source_boundary_ready"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
	}{
		Schema:                  gate.Schema,
		Status:                  gate.Status,
		Action:                  gate.Action,
		SourceReport:            gate.SourceReport,
		SourcePreflightID:       gate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        gate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     gate.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        gate.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                gate.CausalID,
		GateHash:                gate.GateHash,
		ReadBackHash:            gate.ReadBackHash,
		Ready:                   gate.WeightedAdmissionResonanceGraftGateReady,
		ReceiptShape:            gate.ReceiptShape,
		GateKind:                gate.GateKind,
		GateMode:                gate.GateMode,
		GateStage:               gate.GateStage,
		BoundaryVerified:        gate.BoundaryVerified,
		AdmissionRequired:       gate.AdmissionRequired,
		ShadowOnly:              gate.ShadowOnly,
		GraftAllowed:            gate.GraftAllowed,
		DryRunOnly:              gate.DryRunOnly,
		RawDreamTextAllowed:     gate.RawDreamTextAllowed,
		JanusSurfaceAllowed:     gate.JanusSurfaceAllowed,
		CoocLearningAllowed:     gate.CoocLearningAllowed,
		DeltaHarvestAllowed:     gate.DeltaHarvestAllowed,
		BodyMutationAllowed:     gate.BodyMutationAllowed,
		RollbackRequired:        gate.RollbackRequired,
		LiveReady:               gate.LiveReady,
		ContractsReady:          gate.ContractsReady,
		BodyTarget:              gate.BodyTarget,
		WriteAllowed:            gate.WriteAllowed,
		AdmissionAllowed:        gate.AdmissionAllowed,
		LiveAdmissionEnabled:    gate.LiveAdmissionEnabled,
		MutatesState:            gate.MutatesState,
		NextStepBlockedWithout:  gate.NextStepBlockedWithoutResonanceGraftGate,
		SourcePreflightReady:    gate.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     gate.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  gate.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     gate.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       gate.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    gate.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         gate.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       gate.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: gate.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft gate decode failed: %w", err)
	}
	return report, root, nil
}
