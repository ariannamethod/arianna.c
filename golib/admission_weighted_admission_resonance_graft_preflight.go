package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema = "arianna.live_route_weighted_admission_resonance_graft_preflight.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport struct {
	Schema                                              string `json:"schema"`
	Status                                              string `json:"status"`
	Target                                              string `json:"target"`
	TargetKind                                          string `json:"target_kind"`
	TargetMode                                          string `json:"target_mode"`
	Action                                              string `json:"action"`
	WeightedAdmissionResonanceGraftPreflightReady       bool   `json:"weighted_admission_resonance_graft_preflight_ready"`
	WeightedAdmissionResonanceGraftBoundaryConsumed     bool   `json:"weighted_admission_resonance_graft_boundary_consumed"`
	WeightedAdmissionResonanceGraftBoundaryRequired     bool   `json:"weighted_admission_resonance_graft_boundary_required"`
	NextStepBlockedWithoutResonanceGraftPreflight       bool   `json:"next_step_blocked_without_resonance_graft_preflight"`
	WeightedAdmissionResonanceGraftPreflightID          string `json:"weighted_admission_resonance_graft_preflight_id"`
	ReceiptShape                                        string `json:"receipt_shape"`
	PreflightKind                                       string `json:"preflight_kind"`
	PreflightMode                                       string `json:"preflight_mode"`
	PreflightStage                                      string `json:"preflight_stage"`
	CausalID                                            string `json:"causal_id"`
	PreflightHash                                       string `json:"preflight_hash"`
	ReadBackHash                                        string `json:"read_back_hash"`
	BoundaryVerified                                    bool   `json:"boundary_verified"`
	ObservationVerified                                 bool   `json:"observation_verified"`
	ReceiverVerified                                    bool   `json:"receiver_verified"`
	IntentVerified                                      bool   `json:"intent_verified"`
	FinalGateVerified                                   bool   `json:"final_gate_verified"`
	SealVerified                                        bool   `json:"seal_verified"`
	PermitVerified                                      bool   `json:"permit_verified"`
	AuthorityVerified                                   bool   `json:"authority_verified"`
	AdmissionRequired                                   bool   `json:"admission_required"`
	ShadowOnly                                          bool   `json:"shadow_only"`
	GraftAllowed                                        bool   `json:"graft_allowed"`
	DryRunOnly                                          bool   `json:"dry_run_only"`
	LiveReady                                           bool   `json:"live_ready"`
	RawDreamTextAllowed                                 bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                               bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                 bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                 bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                 bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                 bool   `json:"body_mutation_allowed"`
	RollbackRequired                                    bool   `json:"rollback_required"`
	SourceSchema                                        string `json:"source_schema"`
	SourceStatus                                        string `json:"source_status"`
	SourceTarget                                        string `json:"source_target"`
	SourceReport                                        string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftBoundaryID     string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady  bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryCausal string `json:"source_weighted_admission_resonance_graft_boundary_causal_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryHash   string `json:"source_weighted_admission_resonance_graft_boundary_hash"`
	SourceWeightedAdmissionResonanceGraftBoundaryRead   string `json:"source_weighted_admission_resonance_graft_boundary_read_back_hash"`
	SourceBoundaryAction                                string `json:"source_boundary_action"`
	SourceBoundaryReceiptShape                          string `json:"source_boundary_receipt_shape"`
	SourceBoundaryKind                                  string `json:"source_boundary_kind"`
	SourceBoundaryMode                                  string `json:"source_boundary_mode"`
	SourceBoundaryStage                                 string `json:"source_boundary_stage"`
	SourceBoundaryShadowOnly                            bool   `json:"source_boundary_shadow_only"`
	SourceBoundaryGraftAllowed                          bool   `json:"source_boundary_graft_allowed"`
	SourceBoundaryDryRunOnly                            bool   `json:"source_boundary_dry_run_only"`
	SourceBoundaryLiveReady                             bool   `json:"source_boundary_live_ready"`
	SourceBoundaryRawDreamTextAllowed                   bool   `json:"source_boundary_raw_dream_text_allowed"`
	SourceBoundaryRawDreamTextObserved                  bool   `json:"source_boundary_raw_dream_text_observed"`
	SourceBoundaryRawDreamTextForwarded                 bool   `json:"source_boundary_raw_dream_text_forwarded"`
	SourceBoundaryJanusSurfaceAllowed                   bool   `json:"source_boundary_janus_surface_allowed"`
	SourceBoundaryCoocLearningAllowed                   bool   `json:"source_boundary_cooc_learning_allowed"`
	SourceBoundaryDeltaHarvestAllowed                   bool   `json:"source_boundary_delta_harvest_allowed"`
	SourceBoundaryBodyMutationAllowed                   bool   `json:"source_boundary_body_mutation_allowed"`
	SourceBoundaryRollbackRequired                      bool   `json:"source_boundary_rollback_required"`
	SourceObservationSchema                             string `json:"source_observation_schema"`
	SourceObservationStatus                             string `json:"source_observation_status"`
	SourceObservationTarget                             string `json:"source_observation_target"`
	SourceObservationReport                             string `json:"source_observation_report"`
	SourceWeightedAdmissionResonanceObservationID       string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady    bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceObservationCausal   string `json:"source_weighted_admission_resonance_observation_causal_id"`
	SourceWeightedAdmissionResonanceObservationAppend   string `json:"source_weighted_admission_resonance_observation_append_hash"`
	SourceWeightedAdmissionResonanceObservationRead     string `json:"source_weighted_admission_resonance_observation_read_back_hash"`
	SourceObserver                                      string `json:"source_observer"`
	SourceObserverKind                                  string `json:"source_observer_kind"`
	SourceObservationKind                               string `json:"source_observation_kind"`
	SourceObservationMode                               string `json:"source_observation_mode"`
	SourceAppendOnly                                    bool   `json:"source_append_only"`
	SourceReadBack                                      bool   `json:"source_read_back"`
	SourceReceiptVerified                               bool   `json:"source_receipt_verified"`
	SourceDryRunOnly                                    bool   `json:"source_dry_run_only"`
	SourceObservationRawDreamTextObserved               bool   `json:"source_observation_raw_dream_text_observed"`
	SourceObservationRawDreamTextForwarded              bool   `json:"source_observation_raw_dream_text_forwarded"`
	SourceObservationJanusSurfaceAllowed                bool   `json:"source_observation_janus_surface_allowed"`
	SourceObservationCoocLearningAllowed                bool   `json:"source_observation_cooc_learning_allowed"`
	SourceObservationDeltaHarvestAllowed                bool   `json:"source_observation_delta_harvest_allowed"`
	SourceObservationBodyMutationAllowed                bool   `json:"source_observation_body_mutation_allowed"`
	SourceObservationRollbackRequired                   bool   `json:"source_observation_rollback_required"`
	SourceResonanceReceiverReport                       string `json:"source_resonance_receiver_report"`
	SourceResonanceIntentReport                         string `json:"source_resonance_intent_report"`
	SourceFinalGateReport                               string `json:"source_final_gate_report"`
	SourceSealReport                                    string `json:"source_seal_report"`
	SourcePermitReport                                  string `json:"source_permit_report"`
	SourceAuthorityReport                               string `json:"source_authority_report"`
	SourceContractReport                                string `json:"source_contract_report"`
	SourcePreconditionReport                            string `json:"source_precondition_report"`
	SourceReadinessReport                               string `json:"source_readiness_report"`
	SourceBodyWorkdir                                   string `json:"source_body_workdir"`
	SourceBoundaryReport                                string `json:"source_boundary_report"`
	SourceProofLog                                      string `json:"source_proof_log"`
	SourceFinalGateLog                                  string `json:"source_final_gate_log"`
	SourceWeightedAdmissionResonanceReceiverID          string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady       bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceReceiverCausal      string `json:"source_weighted_admission_resonance_receiver_causal_id"`
	SourceReceiverPreStateHash                          string `json:"source_receiver_pre_state_hash"`
	SourceReceiverPostStateHash                         string `json:"source_receiver_post_state_hash"`
	SourceReceiverStateDeltaHash                        string `json:"source_receiver_state_delta_hash"`
	SourceWeightedAdmissionResonanceIntentConsumed      bool   `json:"source_weighted_admission_resonance_intent_consumed"`
	SourceWeightedAdmissionResonanceIntentRequired      bool   `json:"source_weighted_admission_resonance_intent_required"`
	SourceWeightedAdmissionResonanceIntentReady         bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateConsumed            bool   `json:"source_weighted_admission_final_gate_consumed"`
	SourceWeightedAdmissionFinalGateRequired            bool   `json:"source_weighted_admission_final_gate_required"`
	SourceWeightedAdmissionFinalGateReady               bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed                 bool   `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired                 bool   `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady                    bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed               bool   `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired               bool   `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady                  bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed            bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired            bool   `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested                         bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                              bool   `json:"source_permit_key_matched"`
	SourceRawDreamTextAllowed                           bool   `json:"source_raw_dream_text_allowed"`
	SourceRawDreamTextObserved                          bool   `json:"source_raw_dream_text_observed"`
	SourceRawDreamTextForwarded                         bool   `json:"source_raw_dream_text_forwarded"`
	SourceJanusSurfaceAllowed                           bool   `json:"source_janus_surface_allowed"`
	SourceCoocLearningAllowed                           bool   `json:"source_cooc_learning_allowed"`
	SourceDeltaHarvestAllowed                           bool   `json:"source_delta_harvest_allowed"`
	SourceBodyMutationAllowed                           bool   `json:"source_body_mutation_allowed"`
	SourceRollbackRequired                              bool   `json:"source_rollback_required"`
	SourcePreStateHashRequired                          bool   `json:"source_pre_state_hash_required"`
	SourcePostStateHashRequired                         bool   `json:"source_post_state_hash_required"`
	BodySmokeWeighted                                   bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                    bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                 bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                        bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                             bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                              bool   `json:"source_authority_granted"`
	AuthorityGranted                                    bool   `json:"authority_granted"`
	ContractsReady                                      bool   `json:"contracts_ready"`
	WriteAllowed                                        bool   `json:"write_allowed"`
	AdmissionAllowed                                    bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                bool   `json:"live_admission_enabled"`
	MutatesState                                        bool   `json:"mutates_state"`
	BodyTarget                                          string `json:"body_target"`
	Passed                                              bool   `json:"passed"`
	Reason                                              string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflight(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-preflight RESONANCE_GRAFT_BOUNDARY_REPORT RESONANCE_GRAFT_PREFLIGHT_REPORT")
	}
	boundaryPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft preflight output path missing")
	}
	boundary, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportForAssert(boundaryPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportError(boundary, root); err != nil {
		return err
	}
	preflight := admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport{
		Schema:     admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema,
		Status:     "shadow_graft_preflight_ready_dry_run",
		Target:     "resonance",
		TargetKind: "weighted_internal_world_shadow_graft_preflight",
		TargetMode: "receipt_only_closed_preflight_dry_run",
		Action:     "prepare_weighted_resonance_shadow_graft_preflight_dry_run",
		WeightedAdmissionResonanceGraftPreflightReady:   true,
		WeightedAdmissionResonanceGraftBoundaryConsumed: true,
		WeightedAdmissionResonanceGraftBoundaryRequired: true,
		NextStepBlockedWithoutResonanceGraftPreflight:   true,
		ReceiptShape:     "weighted_resonance_shadow_graft_preflight_contract",
		PreflightKind:    "shadow_graft_preflight",
		PreflightMode:    "no_mutation_preflight",
		PreflightStage:   "pre_live_graft_admission",
		BoundaryVerified: true,
		ObservationVerified: boundary.SourceWeightedAdmissionResonanceObservationReady &&
			boundary.SourceReceiptVerified,
		ReceiverVerified:      boundary.SourceWeightedAdmissionResonanceReceiverReady,
		IntentVerified:        boundary.SourceWeightedAdmissionResonanceIntentReady,
		FinalGateVerified:     boundary.SourceWeightedAdmissionFinalGateReady,
		SealVerified:          boundary.SourceWeightedAdmissionSealReady,
		PermitVerified:        boundary.SourceWeightedAdmissionPermitReady,
		AuthorityVerified:     boundary.SourceWeightedAdmissionAuthorityConsumed && boundary.SourceWeightedAdmissionAuthorityRequired,
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
		SourceSchema:          boundary.Schema,
		SourceStatus:          boundary.Status,
		SourceTarget:          boundary.Target,
		SourceReport:          boundaryPath,
		SourceWeightedAdmissionResonanceGraftBoundaryID:     boundary.WeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:  boundary.WeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceGraftBoundaryCausal: boundary.CausalID,
		SourceWeightedAdmissionResonanceGraftBoundaryHash:   boundary.BoundaryHash,
		SourceWeightedAdmissionResonanceGraftBoundaryRead:   boundary.ReadBackHash,
		SourceBoundaryAction:                                boundary.Action,
		SourceBoundaryReceiptShape:                          boundary.ReceiptShape,
		SourceBoundaryKind:                                  boundary.BoundaryKind,
		SourceBoundaryMode:                                  boundary.BoundaryMode,
		SourceBoundaryStage:                                 boundary.BoundaryStage,
		SourceBoundaryShadowOnly:                            boundary.ShadowOnly,
		SourceBoundaryGraftAllowed:                          boundary.GraftAllowed,
		SourceBoundaryDryRunOnly:                            boundary.DryRunOnly,
		SourceBoundaryLiveReady:                             boundary.LiveReady,
		SourceBoundaryRawDreamTextAllowed:                   boundary.RawDreamTextAllowed,
		SourceBoundaryRawDreamTextObserved:                  boundary.RawDreamTextObserved,
		SourceBoundaryRawDreamTextForwarded:                 boundary.RawDreamTextForwarded,
		SourceBoundaryJanusSurfaceAllowed:                   boundary.JanusSurfaceAllowed,
		SourceBoundaryCoocLearningAllowed:                   boundary.CoocLearningAllowed,
		SourceBoundaryDeltaHarvestAllowed:                   boundary.DeltaHarvestAllowed,
		SourceBoundaryBodyMutationAllowed:                   boundary.BodyMutationAllowed,
		SourceBoundaryRollbackRequired:                      boundary.RollbackRequired,
		SourceObservationSchema:                             boundary.SourceSchema,
		SourceObservationStatus:                             boundary.SourceStatus,
		SourceObservationTarget:                             boundary.SourceTarget,
		SourceObservationReport:                             boundary.SourceReport,
		SourceWeightedAdmissionResonanceObservationID:       boundary.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:    boundary.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceObservationCausal:   boundary.SourceWeightedAdmissionResonanceObservationCausal,
		SourceWeightedAdmissionResonanceObservationAppend:   boundary.SourceWeightedAdmissionResonanceObservationAppend,
		SourceWeightedAdmissionResonanceObservationRead:     boundary.SourceWeightedAdmissionResonanceObservationRead,
		SourceObserver:                                      boundary.SourceObserver,
		SourceObserverKind:                                  boundary.SourceObserverKind,
		SourceObservationKind:                               boundary.SourceObservationKind,
		SourceObservationMode:                               boundary.SourceObservationMode,
		SourceAppendOnly:                                    boundary.SourceAppendOnly,
		SourceReadBack:                                      boundary.SourceReadBack,
		SourceReceiptVerified:                               boundary.SourceReceiptVerified,
		SourceDryRunOnly:                                    boundary.SourceDryRunOnly,
		SourceObservationRawDreamTextObserved:               boundary.SourceObservationRawDreamTextObserved,
		SourceObservationRawDreamTextForwarded:              boundary.SourceObservationRawDreamTextForwarded,
		SourceObservationJanusSurfaceAllowed:                boundary.SourceObservationJanusSurfaceAllowed,
		SourceObservationCoocLearningAllowed:                boundary.SourceObservationCoocLearningAllowed,
		SourceObservationDeltaHarvestAllowed:                boundary.SourceObservationDeltaHarvestAllowed,
		SourceObservationBodyMutationAllowed:                boundary.SourceObservationBodyMutationAllowed,
		SourceObservationRollbackRequired:                   boundary.SourceObservationRollbackRequired,
		SourceResonanceReceiverReport:                       boundary.SourceResonanceReceiverReport,
		SourceResonanceIntentReport:                         boundary.SourceResonanceIntentReport,
		SourceFinalGateReport:                               boundary.SourceFinalGateReport,
		SourceSealReport:                                    boundary.SourceSealReport,
		SourcePermitReport:                                  boundary.SourcePermitReport,
		SourceAuthorityReport:                               boundary.SourceAuthorityReport,
		SourceContractReport:                                boundary.SourceContractReport,
		SourcePreconditionReport:                            boundary.SourcePreconditionReport,
		SourceReadinessReport:                               boundary.SourceReadinessReport,
		SourceBodyWorkdir:                                   boundary.SourceBodyWorkdir,
		SourceBoundaryReport:                                boundary.SourceBoundaryReport,
		SourceProofLog:                                      boundary.SourceProofLog,
		SourceFinalGateLog:                                  boundary.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceReceiverID:          boundary.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:       boundary.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceReceiverCausal:      boundary.SourceWeightedAdmissionResonanceReceiverCausal,
		SourceReceiverPreStateHash:                          boundary.SourceReceiverPreStateHash,
		SourceReceiverPostStateHash:                         boundary.SourceReceiverPostStateHash,
		SourceReceiverStateDeltaHash:                        boundary.SourceReceiverStateDeltaHash,
		SourceWeightedAdmissionResonanceIntentConsumed:      boundary.SourceWeightedAdmissionResonanceIntentConsumed,
		SourceWeightedAdmissionResonanceIntentRequired:      boundary.SourceWeightedAdmissionResonanceIntentRequired,
		SourceWeightedAdmissionResonanceIntentReady:         boundary.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:            boundary.SourceWeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:            boundary.SourceWeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:               boundary.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:                 boundary.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:                 boundary.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:                    boundary.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:               boundary.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:               boundary.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:                  boundary.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:            boundary.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:            boundary.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                         boundary.SourceManualPermitRequested,
		SourcePermitKeyMatched:                              boundary.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                           boundary.SourceRawDreamTextAllowed,
		SourceRawDreamTextObserved:                          boundary.SourceRawDreamTextObserved,
		SourceRawDreamTextForwarded:                         boundary.SourceRawDreamTextForwarded,
		SourceJanusSurfaceAllowed:                           boundary.SourceJanusSurfaceAllowed,
		SourceCoocLearningAllowed:                           boundary.SourceCoocLearningAllowed,
		SourceDeltaHarvestAllowed:                           boundary.SourceDeltaHarvestAllowed,
		SourceBodyMutationAllowed:                           boundary.SourceBodyMutationAllowed,
		SourceRollbackRequired:                              boundary.SourceRollbackRequired,
		SourcePreStateHashRequired:                          boundary.SourcePreStateHashRequired,
		SourcePostStateHashRequired:                         boundary.SourcePostStateHashRequired,
		BodySmokeWeighted:                                   boundary.BodySmokeWeighted,
		NanoDirectRunner:                                    boundary.NanoDirectRunner,
		NanoDirectFinalGate:                                 boundary.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                        boundary.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                             boundary.BoundaryReportFullChain,
		SourceAuthorityGranted:                              boundary.SourceAuthorityGranted,
		AuthorityGranted:                                    false,
		ContractsReady:                                      false,
		WriteAllowed:                                        false,
		AdmissionAllowed:                                    false,
		LiveAdmissionEnabled:                                false,
		MutatesState:                                        false,
		BodyTarget:                                          "none",
		Passed:                                              true,
		Reason:                                              "weighted resonance shadow graft preflight prepared without body mutation",
	}
	preflight.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftPreflightCausalID(preflight)
	preflight.PreflightHash = admissionLiveRouteWeightedAdmissionResonanceGraftPreflightHash(preflight)
	preflight.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReadBackHash(preflight)
	preflight.WeightedAdmissionResonanceGraftPreflightID = admissionLiveRouteWeightedAdmissionResonanceGraftPreflightID(preflight)
	if preflight.CausalID == "" ||
		preflight.PreflightHash == "" ||
		preflight.ReadBackHash == "" ||
		preflight.WeightedAdmissionResonanceGraftPreflightID == "" ||
		preflight.PreflightHash == preflight.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft preflight read-back proof failed")
	}
	raw, err := json.MarshalIndent(preflight, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft preflight marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft preflight write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-preflight] pass: resonance_graft_preflight_report=%s resonance_graft_boundary_report=%s\n", outputPath, boundaryPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-preflight-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft preflight schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft preflight schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema)
	}
	if report.Status != "shadow_graft_preflight_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight status mismatch: got %q want %q", report.Status, "shadow_graft_preflight_ready_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft preflight target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_preflight" {
		return fmt.Errorf("weighted admission resonance graft preflight target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_preflight")
	}
	if report.TargetMode != "receipt_only_closed_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_preflight_dry_run")
	}
	if report.Action != "prepare_weighted_resonance_shadow_graft_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight action mismatch: got %q want %q", report.Action, "prepare_weighted_resonance_shadow_graft_preflight_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_preflight_contract" {
		return fmt.Errorf("weighted admission resonance graft preflight receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_preflight_contract")
	}
	if report.PreflightKind != "shadow_graft_preflight" {
		return fmt.Errorf("weighted admission resonance graft preflight preflight_kind mismatch: got %q want %q", report.PreflightKind, "shadow_graft_preflight")
	}
	if report.PreflightMode != "no_mutation_preflight" {
		return fmt.Errorf("weighted admission resonance graft preflight preflight_mode mismatch: got %q want %q", report.PreflightMode, "no_mutation_preflight")
	}
	if report.PreflightStage != "pre_live_graft_admission" {
		return fmt.Errorf("weighted admission resonance graft preflight preflight_stage mismatch: got %q want %q", report.PreflightStage, "pre_live_graft_admission")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_preflight_ready", report.WeightedAdmissionResonanceGraftPreflightReady},
		{"weighted_admission_resonance_graft_boundary_consumed", report.WeightedAdmissionResonanceGraftBoundaryConsumed},
		{"weighted_admission_resonance_graft_boundary_required", report.WeightedAdmissionResonanceGraftBoundaryRequired},
		{"next_step_blocked_without_resonance_graft_preflight", report.NextStepBlockedWithoutResonanceGraftPreflight},
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
			return fmt.Errorf("weighted admission resonance graft preflight %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft preflight opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
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
			return fmt.Errorf("weighted admission resonance graft preflight %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft preflight source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema)
	}
	if report.SourceStatus != "shadow_graft_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_boundary_declared_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft preflight source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceBoundaryAction != "declare_weighted_resonance_shadow_graft_boundary_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight source_boundary_action mismatch: got %q want %q", report.SourceBoundaryAction, "declare_weighted_resonance_shadow_graft_boundary_dry_run")
	}
	if report.SourceBoundaryReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" {
		return fmt.Errorf("weighted admission resonance graft preflight source_boundary_receipt_shape mismatch: got %q want %q", report.SourceBoundaryReceiptShape, "weighted_resonance_observation_shadow_graft_boundary")
	}
	if report.SourceBoundaryKind != "shadow_graft_boundary" ||
		report.SourceBoundaryMode != "no_mutation_receipt" ||
		report.SourceBoundaryStage != "pre_live_graft" {
		return fmt.Errorf("weighted admission resonance graft preflight source boundary shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft preflight source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryCausal, "weighted-resonance-graft-boundary-causal-") {
		return fmt.Errorf("weighted admission resonance graft preflight source boundary causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryHash, "weighted-resonance-graft-boundary-") {
		return fmt.Errorf("weighted admission resonance graft preflight source boundary hash prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryRead, "weighted-resonance-graft-boundary-read-") ||
		report.SourceWeightedAdmissionResonanceGraftBoundaryHash == report.SourceWeightedAdmissionResonanceGraftBoundaryRead {
		return fmt.Errorf("weighted admission resonance graft preflight source boundary read-back mismatch")
	}
	if report.SourceObservationSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema {
		return fmt.Errorf("weighted admission resonance graft preflight source_observation_schema mismatch: got %q want %q", report.SourceObservationSchema, admissionLiveRouteWeightedAdmissionResonanceObservationSchema)
	}
	if report.SourceObservationStatus != "observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft preflight source_observation_status mismatch: got %q want %q", report.SourceObservationStatus, "observation_recorded_dry_run")
	}
	if report.SourceObservationTarget != "resonance" ||
		report.SourceObserver != "resonance" ||
		report.SourceObserverKind != "internal_world" ||
		report.SourceObservationKind != "weighted_receiver_state_proof" ||
		report.SourceObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft preflight source observation shape mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft preflight source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") {
		return fmt.Errorf("weighted admission resonance graft preflight source observation causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") {
		return fmt.Errorf("weighted admission resonance graft preflight source observation append prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") {
		return fmt.Errorf("weighted admission resonance graft preflight source observation read-back prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft preflight source receiver id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") {
		return fmt.Errorf("weighted admission resonance graft preflight source receiver causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(report.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(report.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		report.SourceReceiverPreStateHash == report.SourceReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft preflight source receiver state proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft preflight body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft preflight causal_id mismatch")
	}
	if report.PreflightHash == "" || report.PreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightHash(report) {
		return fmt.Errorf("weighted admission resonance graft preflight preflight_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft preflight read_back_hash mismatch")
	}
	if report.PreflightHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft preflight read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftPreflightID == "" || report.WeightedAdmissionResonanceGraftPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftPreflightID(report) {
		return fmt.Errorf("weighted admission resonance graft preflight id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft preflight prepared without body mutation" {
		return fmt.Errorf("weighted admission resonance graft preflight reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftPreflightCausalID(preflight admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport) string {
	h := hashJSON(struct {
		SourceBoundaryID           string `json:"source_boundary_id"`
		SourceBoundaryReadBackHash string `json:"source_boundary_read_back_hash"`
		SourceObservationID        string `json:"source_observation_id"`
		SourceReceiverID           string `json:"source_receiver_id"`
		Target                     string `json:"target"`
		PreflightKind              string `json:"preflight_kind"`
		PreflightStage             string `json:"preflight_stage"`
	}{
		SourceBoundaryID:           preflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceBoundaryReadBackHash: preflight.SourceWeightedAdmissionResonanceGraftBoundaryRead,
		SourceObservationID:        preflight.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:           preflight.SourceWeightedAdmissionResonanceReceiverID,
		Target:                     preflight.Target,
		PreflightKind:              preflight.PreflightKind,
		PreflightStage:             preflight.PreflightStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-preflight-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftPreflightHash(preflight admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceBoundaryHash     string `json:"source_boundary_hash"`
		SourceBoundaryReadBack string `json:"source_boundary_read_back_hash"`
		PreflightMode          string `json:"preflight_mode"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		DryRunOnly             bool   `json:"dry_run_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
	}{
		CausalID:               preflight.CausalID,
		SourceBoundaryID:       preflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceBoundaryHash:     preflight.SourceWeightedAdmissionResonanceGraftBoundaryHash,
		SourceBoundaryReadBack: preflight.SourceWeightedAdmissionResonanceGraftBoundaryRead,
		PreflightMode:          preflight.PreflightMode,
		AdmissionRequired:      preflight.AdmissionRequired,
		ShadowOnly:             preflight.ShadowOnly,
		DryRunOnly:             preflight.DryRunOnly,
		GraftAllowed:           preflight.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-preflight-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReadBackHash(preflight admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport) string {
	h := hashJSON(struct {
		PreflightHash  string `json:"preflight_hash"`
		SourceBoundary string `json:"source_boundary_id"`
		PreflightKind  string `json:"preflight_kind"`
		PreflightReady bool   `json:"preflight_ready"`
		BodyMutation   bool   `json:"body_mutation"`
		AdmissionOpen  bool   `json:"admission_open"`
	}{
		PreflightHash:  preflight.PreflightHash,
		SourceBoundary: preflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		PreflightKind:  preflight.PreflightKind,
		PreflightReady: preflight.WeightedAdmissionResonanceGraftPreflightReady,
		BodyMutation:   preflight.BodyMutationAllowed,
		AdmissionOpen:  preflight.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-preflight-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftPreflightID(preflight admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		PreflightHash           string `json:"preflight_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		PreflightKind           string `json:"preflight_kind"`
		PreflightMode           string `json:"preflight_mode"`
		PreflightStage          string `json:"preflight_stage"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_preflight"`
		SourceBoundaryReady     bool   `json:"source_boundary_ready"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
	}{
		Schema:                  preflight.Schema,
		Status:                  preflight.Status,
		Action:                  preflight.Action,
		SourceReport:            preflight.SourceReport,
		SourceBoundaryID:        preflight.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     preflight.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        preflight.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                preflight.CausalID,
		PreflightHash:           preflight.PreflightHash,
		ReadBackHash:            preflight.ReadBackHash,
		Ready:                   preflight.WeightedAdmissionResonanceGraftPreflightReady,
		ReceiptShape:            preflight.ReceiptShape,
		PreflightKind:           preflight.PreflightKind,
		PreflightMode:           preflight.PreflightMode,
		PreflightStage:          preflight.PreflightStage,
		BoundaryVerified:        preflight.BoundaryVerified,
		AdmissionRequired:       preflight.AdmissionRequired,
		ShadowOnly:              preflight.ShadowOnly,
		GraftAllowed:            preflight.GraftAllowed,
		DryRunOnly:              preflight.DryRunOnly,
		RawDreamTextAllowed:     preflight.RawDreamTextAllowed,
		JanusSurfaceAllowed:     preflight.JanusSurfaceAllowed,
		CoocLearningAllowed:     preflight.CoocLearningAllowed,
		DeltaHarvestAllowed:     preflight.DeltaHarvestAllowed,
		BodyMutationAllowed:     preflight.BodyMutationAllowed,
		RollbackRequired:        preflight.RollbackRequired,
		LiveReady:               preflight.LiveReady,
		ContractsReady:          preflight.ContractsReady,
		BodyTarget:              preflight.BodyTarget,
		WriteAllowed:            preflight.WriteAllowed,
		AdmissionAllowed:        preflight.AdmissionAllowed,
		LiveAdmissionEnabled:    preflight.LiveAdmissionEnabled,
		MutatesState:            preflight.MutatesState,
		NextStepBlockedWithout:  preflight.NextStepBlockedWithoutResonanceGraftPreflight,
		SourceBoundaryReady:     preflight.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  preflight.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     preflight.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       preflight.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    preflight.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         preflight.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       preflight.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: preflight.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-preflight-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftPreflightReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftPreflightReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft preflight path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft preflight not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft preflight not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft preflight JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft preflight decode failed: %w", err)
	}
	return report, root, nil
}
