package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema = "arianna.live_route_weighted_admission_resonance_graft_candidate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport struct {
	Schema                                               string `json:"schema"`
	Status                                               string `json:"status"`
	Target                                               string `json:"target"`
	TargetKind                                           string `json:"target_kind"`
	TargetMode                                           string `json:"target_mode"`
	Action                                               string `json:"action"`
	WeightedAdmissionResonanceGraftCandidateReady        bool   `json:"weighted_admission_resonance_graft_candidate_ready"`
	WeightedAdmissionResonanceGraftGateConsumed          bool   `json:"weighted_admission_resonance_graft_gate_consumed"`
	WeightedAdmissionResonanceGraftGateRequired          bool   `json:"weighted_admission_resonance_graft_gate_required"`
	NextStepBlockedWithoutResonanceGraftCandidate        bool   `json:"next_step_blocked_without_resonance_graft_candidate"`
	WeightedAdmissionResonanceGraftCandidateID           string `json:"weighted_admission_resonance_graft_candidate_id"`
	ReceiptShape                                         string `json:"receipt_shape"`
	CandidateKind                                        string `json:"candidate_kind"`
	CandidateMode                                        string `json:"candidate_mode"`
	CandidateStage                                       string `json:"candidate_stage"`
	CausalID                                             string `json:"causal_id"`
	CandidateHash                                        string `json:"candidate_hash"`
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
	SourceWeightedAdmissionResonanceGraftGateID          string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady       bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftGateCausal      string `json:"source_weighted_admission_resonance_graft_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftGateHash        string `json:"source_weighted_admission_resonance_graft_gate_hash"`
	SourceWeightedAdmissionResonanceGraftGateRead        string `json:"source_weighted_admission_resonance_graft_gate_read_back_hash"`
	SourceGateAction                                     string `json:"source_gate_action"`
	SourceGateReceiptShape                               string `json:"source_gate_receipt_shape"`
	SourceGateKind                                       string `json:"source_gate_kind"`
	SourceGateMode                                       string `json:"source_gate_mode"`
	SourceGateStage                                      string `json:"source_gate_stage"`
	SourceGateShadowOnly                                 bool   `json:"source_gate_shadow_only"`
	SourceGateGraftAllowed                               bool   `json:"source_gate_graft_allowed"`
	SourceGateDryRunOnly                                 bool   `json:"source_gate_dry_run_only"`
	SourceGateLiveReady                                  bool   `json:"source_gate_live_ready"`
	SourceGateRawDreamTextAllowed                        bool   `json:"source_gate_raw_dream_text_allowed"`
	SourceGateRawDreamTextObserved                       bool   `json:"source_gate_raw_dream_text_observed"`
	SourceGateRawDreamTextForwarded                      bool   `json:"source_gate_raw_dream_text_forwarded"`
	SourceGateJanusSurfaceAllowed                        bool   `json:"source_gate_janus_surface_allowed"`
	SourceGateCoocLearningAllowed                        bool   `json:"source_gate_cooc_learning_allowed"`
	SourceGateDeltaHarvestAllowed                        bool   `json:"source_gate_delta_harvest_allowed"`
	SourceGateBodyMutationAllowed                        bool   `json:"source_gate_body_mutation_allowed"`
	SourceGateRollbackRequired                           bool   `json:"source_gate_rollback_required"`
	SourceGateNextStepBlockedWithoutResonanceGraftGate   bool   `json:"source_next_step_blocked_without_resonance_graft_gate"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate RESONANCE_GRAFT_GATE_REPORT RESONANCE_GRAFT_CANDIDATE_REPORT")
	}
	gatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft candidate output path missing")
	}
	gate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftGateReportForAssert(gatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftGateReportError(gate, root); err != nil {
		return err
	}
	candidate := admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport{
		Schema:     admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema,
		Status:     "shadow_graft_candidate_ready_dry_run",
		Target:     "resonance",
		TargetKind: "weighted_internal_world_shadow_graft_candidate",
		TargetMode: "receipt_only_closed_candidate_dry_run",
		Action:     "draft_weighted_resonance_shadow_graft_candidate_dry_run",
		WeightedAdmissionResonanceGraftCandidateReady: true,
		WeightedAdmissionResonanceGraftGateConsumed:   true,
		WeightedAdmissionResonanceGraftGateRequired:   true,
		NextStepBlockedWithoutResonanceGraftCandidate: true,
		ReceiptShape:          "weighted_resonance_shadow_graft_candidate_contract",
		CandidateKind:         "shadow_graft_candidate",
		CandidateMode:         "no_mutation_candidate",
		CandidateStage:        "pre_live_graft_candidate",
		PreflightVerified:     gate.PreflightVerified,
		BoundaryVerified:      gate.BoundaryVerified,
		ObservationVerified:   gate.ObservationVerified,
		ReceiverVerified:      gate.ReceiverVerified,
		IntentVerified:        gate.IntentVerified,
		FinalGateVerified:     gate.FinalGateVerified,
		SealVerified:          gate.SealVerified,
		PermitVerified:        gate.PermitVerified,
		AuthorityVerified:     gate.AuthorityVerified,
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
		SourceSchema:          gate.Schema,
		SourceStatus:          gate.Status,
		SourceTarget:          gate.Target,
		SourceReport:          gatePath,
		SourceWeightedAdmissionResonanceGraftGateID:     gate.WeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:  gate.WeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftGateCausal: gate.CausalID,
		SourceWeightedAdmissionResonanceGraftGateHash:   gate.GateHash,
		SourceWeightedAdmissionResonanceGraftGateRead:   gate.ReadBackHash,
		SourceGateAction:                                     gate.Action,
		SourceGateReceiptShape:                               gate.ReceiptShape,
		SourceGateKind:                                       gate.GateKind,
		SourceGateMode:                                       gate.GateMode,
		SourceGateStage:                                      gate.GateStage,
		SourceGateShadowOnly:                                 gate.ShadowOnly,
		SourceGateGraftAllowed:                               gate.GraftAllowed,
		SourceGateDryRunOnly:                                 gate.DryRunOnly,
		SourceGateLiveReady:                                  gate.LiveReady,
		SourceGateRawDreamTextAllowed:                        gate.RawDreamTextAllowed,
		SourceGateRawDreamTextObserved:                       gate.RawDreamTextObserved,
		SourceGateRawDreamTextForwarded:                      gate.RawDreamTextForwarded,
		SourceGateJanusSurfaceAllowed:                        gate.JanusSurfaceAllowed,
		SourceGateCoocLearningAllowed:                        gate.CoocLearningAllowed,
		SourceGateDeltaHarvestAllowed:                        gate.DeltaHarvestAllowed,
		SourceGateBodyMutationAllowed:                        gate.BodyMutationAllowed,
		SourceGateRollbackRequired:                           gate.RollbackRequired,
		SourceGateNextStepBlockedWithoutResonanceGraftGate:   gate.NextStepBlockedWithoutResonanceGraftGate,
		SourceWeightedAdmissionResonanceGraftPreflightID:     gate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:  gate.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftPreflightCausal: gate.SourceWeightedAdmissionResonanceGraftPreflightCausal,
		SourceWeightedAdmissionResonanceGraftPreflightHash:   gate.SourceWeightedAdmissionResonanceGraftPreflightHash,
		SourceWeightedAdmissionResonanceGraftPreflightRead:   gate.SourceWeightedAdmissionResonanceGraftPreflightRead,
		SourcePreflightAction:                                gate.SourcePreflightAction,
		SourcePreflightReceiptShape:                          gate.SourcePreflightReceiptShape,
		SourcePreflightKind:                                  gate.SourcePreflightKind,
		SourcePreflightMode:                                  gate.SourcePreflightMode,
		SourcePreflightStage:                                 gate.SourcePreflightStage,
		SourcePreflightShadowOnly:                            gate.SourcePreflightShadowOnly,
		SourcePreflightGraftAllowed:                          gate.SourcePreflightGraftAllowed,
		SourcePreflightDryRunOnly:                            gate.SourcePreflightDryRunOnly,
		SourcePreflightLiveReady:                             gate.SourcePreflightLiveReady,
		SourcePreflightRawDreamTextAllowed:                   gate.SourcePreflightRawDreamTextAllowed,
		SourcePreflightRawDreamTextObserved:                  gate.SourcePreflightRawDreamTextObserved,
		SourcePreflightRawDreamTextForwarded:                 gate.SourcePreflightRawDreamTextForwarded,
		SourcePreflightJanusSurfaceAllowed:                   gate.SourcePreflightJanusSurfaceAllowed,
		SourcePreflightCoocLearningAllowed:                   gate.SourcePreflightCoocLearningAllowed,
		SourcePreflightDeltaHarvestAllowed:                   gate.SourcePreflightDeltaHarvestAllowed,
		SourcePreflightBodyMutationAllowed:                   gate.SourcePreflightBodyMutationAllowed,
		SourcePreflightRollbackRequired:                      gate.SourcePreflightRollbackRequired,
		SourceGraftBoundarySchema:                            gate.SourceGraftBoundarySchema,
		SourceGraftBoundaryStatus:                            gate.SourceGraftBoundaryStatus,
		SourceGraftBoundaryTarget:                            gate.SourceGraftBoundaryTarget,
		SourceGraftBoundaryReport:                            gate.SourceGraftBoundaryReport,
		SourceWeightedAdmissionResonanceGraftBoundaryID:      gate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:   gate.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceGraftBoundaryCausal:  gate.SourceWeightedAdmissionResonanceGraftBoundaryCausal,
		SourceWeightedAdmissionResonanceGraftBoundaryHash:    gate.SourceWeightedAdmissionResonanceGraftBoundaryHash,
		SourceWeightedAdmissionResonanceGraftBoundaryRead:    gate.SourceWeightedAdmissionResonanceGraftBoundaryRead,
		SourceBoundaryAction:                                 gate.SourceBoundaryAction,
		SourceBoundaryReceiptShape:                           gate.SourceBoundaryReceiptShape,
		SourceBoundaryKind:                                   gate.SourceBoundaryKind,
		SourceBoundaryMode:                                   gate.SourceBoundaryMode,
		SourceBoundaryStage:                                  gate.SourceBoundaryStage,
		SourceBoundaryShadowOnly:                             gate.SourceBoundaryShadowOnly,
		SourceBoundaryGraftAllowed:                           gate.SourceBoundaryGraftAllowed,
		SourceBoundaryDryRunOnly:                             gate.SourceBoundaryDryRunOnly,
		SourceBoundaryLiveReady:                              gate.SourceBoundaryLiveReady,
		SourceBoundaryRawDreamTextAllowed:                    gate.SourceBoundaryRawDreamTextAllowed,
		SourceBoundaryRawDreamTextObserved:                   gate.SourceBoundaryRawDreamTextObserved,
		SourceBoundaryRawDreamTextForwarded:                  gate.SourceBoundaryRawDreamTextForwarded,
		SourceBoundaryJanusSurfaceAllowed:                    gate.SourceBoundaryJanusSurfaceAllowed,
		SourceBoundaryCoocLearningAllowed:                    gate.SourceBoundaryCoocLearningAllowed,
		SourceBoundaryDeltaHarvestAllowed:                    gate.SourceBoundaryDeltaHarvestAllowed,
		SourceBoundaryBodyMutationAllowed:                    gate.SourceBoundaryBodyMutationAllowed,
		SourceBoundaryRollbackRequired:                       gate.SourceBoundaryRollbackRequired,
		SourceObservationSchema:                              gate.SourceObservationSchema,
		SourceObservationStatus:                              gate.SourceObservationStatus,
		SourceObservationTarget:                              gate.SourceObservationTarget,
		SourceObservationReport:                              gate.SourceObservationReport,
		SourceWeightedAdmissionResonanceObservationID:        gate.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:     gate.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceObservationCausal:    gate.SourceWeightedAdmissionResonanceObservationCausal,
		SourceWeightedAdmissionResonanceObservationAppend:    gate.SourceWeightedAdmissionResonanceObservationAppend,
		SourceWeightedAdmissionResonanceObservationRead:      gate.SourceWeightedAdmissionResonanceObservationRead,
		SourceObserver:                                       gate.SourceObserver,
		SourceObserverKind:                                   gate.SourceObserverKind,
		SourceObservationKind:                                gate.SourceObservationKind,
		SourceObservationMode:                                gate.SourceObservationMode,
		SourceAppendOnly:                                     gate.SourceAppendOnly,
		SourceReadBack:                                       gate.SourceReadBack,
		SourceReceiptVerified:                                gate.SourceReceiptVerified,
		SourceDryRunOnly:                                     gate.SourceDryRunOnly,
		SourceObservationRawDreamTextObserved:                gate.SourceObservationRawDreamTextObserved,
		SourceObservationRawDreamTextForwarded:               gate.SourceObservationRawDreamTextForwarded,
		SourceObservationJanusSurfaceAllowed:                 gate.SourceObservationJanusSurfaceAllowed,
		SourceObservationCoocLearningAllowed:                 gate.SourceObservationCoocLearningAllowed,
		SourceObservationDeltaHarvestAllowed:                 gate.SourceObservationDeltaHarvestAllowed,
		SourceObservationBodyMutationAllowed:                 gate.SourceObservationBodyMutationAllowed,
		SourceObservationRollbackRequired:                    gate.SourceObservationRollbackRequired,
		SourceResonanceReceiverReport:                        gate.SourceResonanceReceiverReport,
		SourceResonanceIntentReport:                          gate.SourceResonanceIntentReport,
		SourceFinalGateReport:                                gate.SourceFinalGateReport,
		SourceSealReport:                                     gate.SourceSealReport,
		SourcePermitReport:                                   gate.SourcePermitReport,
		SourceAuthorityReport:                                gate.SourceAuthorityReport,
		SourceContractReport:                                 gate.SourceContractReport,
		SourcePreconditionReport:                             gate.SourcePreconditionReport,
		SourceReadinessReport:                                gate.SourceReadinessReport,
		SourceBodyWorkdir:                                    gate.SourceBodyWorkdir,
		SourceBoundaryReport:                                 gate.SourceBoundaryReport,
		SourceProofLog:                                       gate.SourceProofLog,
		SourceFinalGateLog:                                   gate.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceReceiverID:           gate.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:        gate.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceReceiverCausal:       gate.SourceWeightedAdmissionResonanceReceiverCausal,
		SourceReceiverPreStateHash:                           gate.SourceReceiverPreStateHash,
		SourceReceiverPostStateHash:                          gate.SourceReceiverPostStateHash,
		SourceReceiverStateDeltaHash:                         gate.SourceReceiverStateDeltaHash,
		SourceWeightedAdmissionResonanceIntentConsumed:       gate.SourceWeightedAdmissionResonanceIntentConsumed,
		SourceWeightedAdmissionResonanceIntentRequired:       gate.SourceWeightedAdmissionResonanceIntentRequired,
		SourceWeightedAdmissionResonanceIntentReady:          gate.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:             gate.SourceWeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:             gate.SourceWeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:                gate.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:                  gate.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:                  gate.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:                     gate.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:                gate.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:                gate.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:                   gate.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:             gate.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:             gate.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                          gate.SourceManualPermitRequested,
		SourcePermitKeyMatched:                               gate.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                            gate.SourceRawDreamTextAllowed,
		SourceRawDreamTextObserved:                           gate.SourceRawDreamTextObserved,
		SourceRawDreamTextForwarded:                          gate.SourceRawDreamTextForwarded,
		SourceJanusSurfaceAllowed:                            gate.SourceJanusSurfaceAllowed,
		SourceCoocLearningAllowed:                            gate.SourceCoocLearningAllowed,
		SourceDeltaHarvestAllowed:                            gate.SourceDeltaHarvestAllowed,
		SourceBodyMutationAllowed:                            gate.SourceBodyMutationAllowed,
		SourceRollbackRequired:                               gate.SourceRollbackRequired,
		SourcePreStateHashRequired:                           gate.SourcePreStateHashRequired,
		SourcePostStateHashRequired:                          gate.SourcePostStateHashRequired,
		BodySmokeWeighted:                                    gate.BodySmokeWeighted,
		NanoDirectRunner:                                     gate.NanoDirectRunner,
		NanoDirectFinalGate:                                  gate.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                         gate.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                              gate.BoundaryReportFullChain,
		SourceAuthorityGranted:                               gate.SourceAuthorityGranted,
		AuthorityGranted:                                     false,
		ContractsReady:                                       false,
		WriteAllowed:                                         false,
		AdmissionAllowed:                                     false,
		LiveAdmissionEnabled:                                 false,
		MutatesState:                                         false,
		BodyTarget:                                           "none",
		Passed:                                               true,
		Reason:                                               "weighted resonance shadow graft candidate drafted without body mutation",
	}
	candidate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateCausalID(candidate)
	candidate.CandidateHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateHash(candidate)
	candidate.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReadBackHash(candidate)
	candidate.WeightedAdmissionResonanceGraftCandidateID = admissionLiveRouteWeightedAdmissionResonanceGraftCandidateID(candidate)
	if candidate.CausalID == "" ||
		candidate.CandidateHash == "" ||
		candidate.ReadBackHash == "" ||
		candidate.WeightedAdmissionResonanceGraftCandidateID == "" ||
		candidate.CandidateHash == candidate.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate read-back proof failed")
	}
	raw, err := json.MarshalIndent(candidate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft candidate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-candidate] pass: resonance_graft_candidate_report=%s resonance_graft_gate_report=%s\n", outputPath, gatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-candidate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft candidate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema {
		return fmt.Errorf("weighted admission resonance graft candidate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema)
	}
	if report.Status != "shadow_graft_candidate_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate status mismatch: got %q want %q", report.Status, "shadow_graft_candidate_ready_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_candidate" {
		return fmt.Errorf("weighted admission resonance graft candidate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_candidate")
	}
	if report.TargetMode != "receipt_only_closed_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_candidate_dry_run")
	}
	if report.Action != "draft_weighted_resonance_shadow_graft_candidate_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate action mismatch: got %q want %q", report.Action, "draft_weighted_resonance_shadow_graft_candidate_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_candidate_contract" {
		return fmt.Errorf("weighted admission resonance graft candidate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_candidate_contract")
	}
	if report.CandidateKind != "shadow_graft_candidate" {
		return fmt.Errorf("weighted admission resonance graft candidate candidate_kind mismatch: got %q want %q", report.CandidateKind, "shadow_graft_candidate")
	}
	if report.CandidateMode != "no_mutation_candidate" {
		return fmt.Errorf("weighted admission resonance graft candidate candidate_mode mismatch: got %q want %q", report.CandidateMode, "no_mutation_candidate")
	}
	if report.CandidateStage != "pre_live_graft_candidate" {
		return fmt.Errorf("weighted admission resonance graft candidate candidate_stage mismatch: got %q want %q", report.CandidateStage, "pre_live_graft_candidate")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_candidate_ready", report.WeightedAdmissionResonanceGraftCandidateReady},
		{"weighted_admission_resonance_graft_gate_consumed", report.WeightedAdmissionResonanceGraftGateConsumed},
		{"weighted_admission_resonance_graft_gate_required", report.WeightedAdmissionResonanceGraftGateRequired},
		{"next_step_blocked_without_resonance_graft_candidate", report.NextStepBlockedWithoutResonanceGraftCandidate},
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
		{"source_weighted_admission_resonance_graft_gate_ready", report.SourceWeightedAdmissionResonanceGraftGateReady},
		{"source_gate_shadow_only", report.SourceGateShadowOnly},
		{"source_gate_dry_run_only", report.SourceGateDryRunOnly},
		{"source_gate_live_ready", report.SourceGateLiveReady},
		{"source_gate_rollback_required", report.SourceGateRollbackRequired},
		{"source_next_step_blocked_without_resonance_graft_gate", report.SourceGateNextStepBlockedWithoutResonanceGraftGate},
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
			return fmt.Errorf("weighted admission resonance graft candidate %s not ready", required.name)
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
		{"source_gate_graft_allowed", report.SourceGateGraftAllowed},
		{"source_gate_raw_dream_text_allowed", report.SourceGateRawDreamTextAllowed},
		{"source_gate_raw_dream_text_observed", report.SourceGateRawDreamTextObserved},
		{"source_gate_raw_dream_text_forwarded", report.SourceGateRawDreamTextForwarded},
		{"source_gate_janus_surface_allowed", report.SourceGateJanusSurfaceAllowed},
		{"source_gate_cooc_learning_allowed", report.SourceGateCoocLearningAllowed},
		{"source_gate_delta_harvest_allowed", report.SourceGateDeltaHarvestAllowed},
		{"source_gate_body_mutation_allowed", report.SourceGateBodyMutationAllowed},
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
			return fmt.Errorf("weighted admission resonance graft candidate opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_gate_id", report.SourceWeightedAdmissionResonanceGraftGateID},
		{"source_weighted_admission_resonance_graft_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftGateCausal},
		{"source_weighted_admission_resonance_graft_gate_hash", report.SourceWeightedAdmissionResonanceGraftGateHash},
		{"source_weighted_admission_resonance_graft_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftGateRead},
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
			return fmt.Errorf("weighted admission resonance graft candidate %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema {
		return fmt.Errorf("weighted admission resonance graft candidate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema)
	}
	if report.SourceStatus != "shadow_graft_gate_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_gate_ready_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceGateAction != "gate_weighted_resonance_shadow_graft_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_gate_action mismatch: got %q want %q", report.SourceGateAction, "gate_weighted_resonance_shadow_graft_dry_run")
	}
	if report.SourceGateReceiptShape != "weighted_resonance_shadow_graft_gate_contract" {
		return fmt.Errorf("weighted admission resonance graft candidate source_gate_receipt_shape mismatch: got %q want %q", report.SourceGateReceiptShape, "weighted_resonance_shadow_graft_gate_contract")
	}
	if report.SourceGateKind != "shadow_graft_gate" ||
		report.SourceGateMode != "no_mutation_gate" ||
		report.SourceGateStage != "pre_live_graft_gate" {
		return fmt.Errorf("weighted admission resonance graft candidate source gate shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateCausal, "weighted-resonance-graft-gate-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate source gate causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateHash, "weighted-resonance-graft-gate-") {
		return fmt.Errorf("weighted admission resonance graft candidate source gate hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateRead, "weighted-resonance-graft-gate-read-") ||
		report.SourceWeightedAdmissionResonanceGraftGateHash == report.SourceWeightedAdmissionResonanceGraftGateRead {
		return fmt.Errorf("weighted admission resonance graft candidate source gate read-back mismatch")
	}
	if report.SourcePreflightAction != "prepare_weighted_resonance_shadow_graft_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_preflight_action mismatch: got %q want %q", report.SourcePreflightAction, "prepare_weighted_resonance_shadow_graft_preflight_dry_run")
	}
	if report.SourcePreflightReceiptShape != "weighted_resonance_shadow_graft_preflight_contract" {
		return fmt.Errorf("weighted admission resonance graft candidate source_preflight_receipt_shape mismatch: got %q want %q", report.SourcePreflightReceiptShape, "weighted_resonance_shadow_graft_preflight_contract")
	}
	if report.SourcePreflightKind != "shadow_graft_preflight" ||
		report.SourcePreflightMode != "no_mutation_preflight" ||
		report.SourcePreflightStage != "pre_live_graft_admission" {
		return fmt.Errorf("weighted admission resonance graft candidate source preflight shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightCausal, "weighted-resonance-graft-preflight-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate source preflight causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightHash, "weighted-resonance-graft-preflight-") {
		return fmt.Errorf("weighted admission resonance graft candidate source preflight hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightRead, "weighted-resonance-graft-preflight-read-") ||
		report.SourceWeightedAdmissionResonanceGraftPreflightHash == report.SourceWeightedAdmissionResonanceGraftPreflightRead {
		return fmt.Errorf("weighted admission resonance graft candidate source preflight read-back mismatch")
	}
	if report.SourceGraftBoundarySchema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft candidate source_graft_boundary_schema mismatch: got %q want %q", report.SourceGraftBoundarySchema, admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema)
	}
	if report.SourceGraftBoundaryStatus != "shadow_graft_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_graft_boundary_status mismatch: got %q want %q", report.SourceGraftBoundaryStatus, "shadow_graft_boundary_declared_dry_run")
	}
	if report.SourceGraftBoundaryTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft candidate source_graft_boundary_target mismatch: got %q want %q", report.SourceGraftBoundaryTarget, "resonance")
	}
	if report.SourceBoundaryAction != "declare_weighted_resonance_shadow_graft_boundary_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_boundary_action mismatch: got %q want %q", report.SourceBoundaryAction, "declare_weighted_resonance_shadow_graft_boundary_dry_run")
	}
	if report.SourceBoundaryReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" {
		return fmt.Errorf("weighted admission resonance graft candidate source_boundary_receipt_shape mismatch: got %q want %q", report.SourceBoundaryReceiptShape, "weighted_resonance_observation_shadow_graft_boundary")
	}
	if report.SourceBoundaryKind != "shadow_graft_boundary" ||
		report.SourceBoundaryMode != "no_mutation_receipt" ||
		report.SourceBoundaryStage != "pre_live_graft" {
		return fmt.Errorf("weighted admission resonance graft candidate source boundary shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft candidate source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryCausal, "weighted-resonance-graft-boundary-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate source boundary causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryHash, "weighted-resonance-graft-boundary-") {
		return fmt.Errorf("weighted admission resonance graft candidate source boundary hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryRead, "weighted-resonance-graft-boundary-read-") ||
		report.SourceWeightedAdmissionResonanceGraftBoundaryHash == report.SourceWeightedAdmissionResonanceGraftBoundaryRead {
		return fmt.Errorf("weighted admission resonance graft candidate source boundary read-back mismatch")
	}
	if report.SourceObservationSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema {
		return fmt.Errorf("weighted admission resonance graft candidate source_observation_schema mismatch: got %q want %q", report.SourceObservationSchema, admissionLiveRouteWeightedAdmissionResonanceObservationSchema)
	}
	if report.SourceObservationStatus != "observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft candidate source_observation_status mismatch: got %q want %q", report.SourceObservationStatus, "observation_recorded_dry_run")
	}
	if report.SourceObservationTarget != "resonance" ||
		report.SourceObserver != "resonance" ||
		report.SourceObserverKind != "internal_world" ||
		report.SourceObservationKind != "weighted_receiver_state_proof" ||
		report.SourceObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft candidate source observation shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft candidate source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate source observation causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") {
		return fmt.Errorf("weighted admission resonance graft candidate source observation append prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") {
		return fmt.Errorf("weighted admission resonance graft candidate source observation read-back prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft candidate source receiver id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") {
		return fmt.Errorf("weighted admission resonance graft candidate source receiver causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(report.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(report.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		report.SourceReceiverPreStateHash == report.SourceReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft candidate source receiver state proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft candidate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate causal_id mismatch")
	}
	if report.CandidateHash == "" || report.CandidateHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate candidate_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft candidate read_back_hash mismatch")
	}
	if report.CandidateHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft candidate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftCandidateID == "" || report.WeightedAdmissionResonanceGraftCandidateID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateID(report) {
		return fmt.Errorf("weighted admission resonance graft candidate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft candidate drafted without body mutation" {
		return fmt.Errorf("weighted admission resonance graft candidate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateCausalID(candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport) string {
	h := hashJSON(struct {
		SourceGateID           string `json:"source_gate_id"`
		SourceGateReadBackHash string `json:"source_gate_read_back_hash"`
		SourcePreflightID      string `json:"source_preflight_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		SourceReceiverID       string `json:"source_receiver_id"`
		Target                 string `json:"target"`
		CandidateKind          string `json:"candidate_kind"`
		CandidateStage         string `json:"candidate_stage"`
	}{
		SourceGateID:           candidate.SourceWeightedAdmissionResonanceGraftGateID,
		SourceGateReadBackHash: candidate.SourceWeightedAdmissionResonanceGraftGateRead,
		SourcePreflightID:      candidate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:       candidate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:    candidate.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:       candidate.SourceWeightedAdmissionResonanceReceiverID,
		Target:                 candidate.Target,
		CandidateKind:          candidate.CandidateKind,
		CandidateStage:         candidate.CandidateStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateHash(candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport) string {
	h := hashJSON(struct {
		CausalID           string `json:"causal_id"`
		SourceGateID       string `json:"source_gate_id"`
		SourceGateHash     string `json:"source_gate_hash"`
		SourceGateReadBack string `json:"source_gate_read_back_hash"`
		CandidateMode      string `json:"candidate_mode"`
		AdmissionRequired  bool   `json:"admission_required"`
		ShadowOnly         bool   `json:"shadow_only"`
		DryRunOnly         bool   `json:"dry_run_only"`
		GraftAllowed       bool   `json:"graft_allowed"`
	}{
		CausalID:           candidate.CausalID,
		SourceGateID:       candidate.SourceWeightedAdmissionResonanceGraftGateID,
		SourceGateHash:     candidate.SourceWeightedAdmissionResonanceGraftGateHash,
		SourceGateReadBack: candidate.SourceWeightedAdmissionResonanceGraftGateRead,
		CandidateMode:      candidate.CandidateMode,
		AdmissionRequired:  candidate.AdmissionRequired,
		ShadowOnly:         candidate.ShadowOnly,
		DryRunOnly:         candidate.DryRunOnly,
		GraftAllowed:       candidate.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReadBackHash(candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport) string {
	h := hashJSON(struct {
		CandidateHash  string `json:"candidate_hash"`
		SourceGate     string `json:"source_gate_id"`
		CandidateKind  string `json:"candidate_kind"`
		CandidateReady bool   `json:"candidate_ready"`
		BodyMutation   bool   `json:"body_mutation"`
		AdmissionOpen  bool   `json:"admission_open"`
	}{
		CandidateHash:  candidate.CandidateHash,
		SourceGate:     candidate.SourceWeightedAdmissionResonanceGraftGateID,
		CandidateKind:  candidate.CandidateKind,
		CandidateReady: candidate.WeightedAdmissionResonanceGraftCandidateReady,
		BodyMutation:   candidate.BodyMutationAllowed,
		AdmissionOpen:  candidate.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftCandidateID(candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourceGateID            string `json:"source_gate_id"`
		SourceGateHash          string `json:"source_gate_hash"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		CandidateHash           string `json:"candidate_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		CandidateKind           string `json:"candidate_kind"`
		CandidateMode           string `json:"candidate_mode"`
		CandidateStage          string `json:"candidate_stage"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_candidate"`
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
		Schema:                  candidate.Schema,
		Status:                  candidate.Status,
		Action:                  candidate.Action,
		SourceReport:            candidate.SourceReport,
		SourceGateID:            candidate.SourceWeightedAdmissionResonanceGraftGateID,
		SourceGateHash:          candidate.SourceWeightedAdmissionResonanceGraftGateHash,
		SourcePreflightID:       candidate.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        candidate.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     candidate.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        candidate.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                candidate.CausalID,
		CandidateHash:           candidate.CandidateHash,
		ReadBackHash:            candidate.ReadBackHash,
		Ready:                   candidate.WeightedAdmissionResonanceGraftCandidateReady,
		ReceiptShape:            candidate.ReceiptShape,
		CandidateKind:           candidate.CandidateKind,
		CandidateMode:           candidate.CandidateMode,
		CandidateStage:          candidate.CandidateStage,
		BoundaryVerified:        candidate.BoundaryVerified,
		AdmissionRequired:       candidate.AdmissionRequired,
		ShadowOnly:              candidate.ShadowOnly,
		GraftAllowed:            candidate.GraftAllowed,
		DryRunOnly:              candidate.DryRunOnly,
		RawDreamTextAllowed:     candidate.RawDreamTextAllowed,
		JanusSurfaceAllowed:     candidate.JanusSurfaceAllowed,
		CoocLearningAllowed:     candidate.CoocLearningAllowed,
		DeltaHarvestAllowed:     candidate.DeltaHarvestAllowed,
		BodyMutationAllowed:     candidate.BodyMutationAllowed,
		RollbackRequired:        candidate.RollbackRequired,
		LiveReady:               candidate.LiveReady,
		ContractsReady:          candidate.ContractsReady,
		BodyTarget:              candidate.BodyTarget,
		WriteAllowed:            candidate.WriteAllowed,
		AdmissionAllowed:        candidate.AdmissionAllowed,
		LiveAdmissionEnabled:    candidate.LiveAdmissionEnabled,
		MutatesState:            candidate.MutatesState,
		NextStepBlockedWithout:  candidate.NextStepBlockedWithoutResonanceGraftCandidate,
		SourcePreflightReady:    candidate.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     candidate.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  candidate.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     candidate.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       candidate.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    candidate.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         candidate.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       candidate.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: candidate.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-candidate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft candidate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft candidate decode failed: %w", err)
	}
	return report, root, nil
}
