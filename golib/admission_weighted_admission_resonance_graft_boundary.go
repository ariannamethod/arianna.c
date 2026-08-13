package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema = "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport struct {
	Schema                                            string `json:"schema"`
	Status                                            string `json:"status"`
	Target                                            string `json:"target"`
	TargetKind                                        string `json:"target_kind"`
	TargetMode                                        string `json:"target_mode"`
	Action                                            string `json:"action"`
	WeightedAdmissionResonanceGraftBoundaryReady      bool   `json:"weighted_admission_resonance_graft_boundary_ready"`
	WeightedAdmissionResonanceObservationConsumed     bool   `json:"weighted_admission_resonance_observation_consumed"`
	WeightedAdmissionResonanceObservationRequired     bool   `json:"weighted_admission_resonance_observation_required"`
	NextStepBlockedWithoutResonanceGraftBoundary      bool   `json:"next_step_blocked_without_resonance_graft_boundary"`
	WeightedAdmissionResonanceGraftBoundaryID         string `json:"weighted_admission_resonance_graft_boundary_id"`
	ReceiptShape                                      string `json:"receipt_shape"`
	BoundaryKind                                      string `json:"boundary_kind"`
	BoundaryMode                                      string `json:"boundary_mode"`
	BoundaryStage                                     string `json:"boundary_stage"`
	CausalID                                          string `json:"causal_id"`
	BoundaryHash                                      string `json:"boundary_hash"`
	ReadBackHash                                      string `json:"read_back_hash"`
	ShadowOnly                                        bool   `json:"shadow_only"`
	GraftAllowed                                      bool   `json:"graft_allowed"`
	DryRunOnly                                        bool   `json:"dry_run_only"`
	LiveReady                                         bool   `json:"live_ready"`
	RawDreamTextAllowed                               bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                              bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                             bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                               bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                               bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                               bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                               bool   `json:"body_mutation_allowed"`
	RollbackRequired                                  bool   `json:"rollback_required"`
	SourceSchema                                      string `json:"source_schema"`
	SourceStatus                                      string `json:"source_status"`
	SourceTarget                                      string `json:"source_target"`
	SourceReport                                      string `json:"source_report"`
	SourceWeightedAdmissionResonanceObservationID     string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady  bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceObservationCausal string `json:"source_weighted_admission_resonance_observation_causal_id"`
	SourceWeightedAdmissionResonanceObservationAppend string `json:"source_weighted_admission_resonance_observation_append_hash"`
	SourceWeightedAdmissionResonanceObservationRead   string `json:"source_weighted_admission_resonance_observation_read_back_hash"`
	SourceObserver                                    string `json:"source_observer"`
	SourceObserverKind                                string `json:"source_observer_kind"`
	SourceObservationKind                             string `json:"source_observation_kind"`
	SourceObservationMode                             string `json:"source_observation_mode"`
	SourceAppendOnly                                  bool   `json:"source_append_only"`
	SourceReadBack                                    bool   `json:"source_read_back"`
	SourceReceiptVerified                             bool   `json:"source_receipt_verified"`
	SourceDryRunOnly                                  bool   `json:"source_dry_run_only"`
	SourceObservationRawDreamTextObserved             bool   `json:"source_observation_raw_dream_text_observed"`
	SourceObservationRawDreamTextForwarded            bool   `json:"source_observation_raw_dream_text_forwarded"`
	SourceObservationJanusSurfaceAllowed              bool   `json:"source_observation_janus_surface_allowed"`
	SourceObservationCoocLearningAllowed              bool   `json:"source_observation_cooc_learning_allowed"`
	SourceObservationDeltaHarvestAllowed              bool   `json:"source_observation_delta_harvest_allowed"`
	SourceObservationBodyMutationAllowed              bool   `json:"source_observation_body_mutation_allowed"`
	SourceObservationRollbackRequired                 bool   `json:"source_observation_rollback_required"`
	SourceResonanceReceiverReport                     string `json:"source_resonance_receiver_report"`
	SourceResonanceIntentReport                       string `json:"source_resonance_intent_report"`
	SourceFinalGateReport                             string `json:"source_final_gate_report"`
	SourceSealReport                                  string `json:"source_seal_report"`
	SourcePermitReport                                string `json:"source_permit_report"`
	SourceAuthorityReport                             string `json:"source_authority_report"`
	SourceContractReport                              string `json:"source_contract_report"`
	SourcePreconditionReport                          string `json:"source_precondition_report"`
	SourceReadinessReport                             string `json:"source_readiness_report"`
	SourceBodyWorkdir                                 string `json:"source_body_workdir"`
	SourceBoundaryReport                              string `json:"source_boundary_report"`
	SourceProofLog                                    string `json:"source_proof_log"`
	SourceFinalGateLog                                string `json:"source_final_gate_log"`
	SourceWeightedAdmissionResonanceReceiverID        string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady     bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceReceiverCausal    string `json:"source_weighted_admission_resonance_receiver_causal_id"`
	SourceReceiverPreStateHash                        string `json:"source_receiver_pre_state_hash"`
	SourceReceiverPostStateHash                       string `json:"source_receiver_post_state_hash"`
	SourceReceiverStateDeltaHash                      string `json:"source_receiver_state_delta_hash"`
	SourceWeightedAdmissionResonanceIntentConsumed    bool   `json:"source_weighted_admission_resonance_intent_consumed"`
	SourceWeightedAdmissionResonanceIntentRequired    bool   `json:"source_weighted_admission_resonance_intent_required"`
	SourceWeightedAdmissionResonanceIntentReady       bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateConsumed          bool   `json:"source_weighted_admission_final_gate_consumed"`
	SourceWeightedAdmissionFinalGateRequired          bool   `json:"source_weighted_admission_final_gate_required"`
	SourceWeightedAdmissionFinalGateReady             bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed               bool   `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired               bool   `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady                  bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed             bool   `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired             bool   `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady                bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed          bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired          bool   `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested                       bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                            bool   `json:"source_permit_key_matched"`
	SourceRawDreamTextAllowed                         bool   `json:"source_raw_dream_text_allowed"`
	SourceRawDreamTextObserved                        bool   `json:"source_raw_dream_text_observed"`
	SourceRawDreamTextForwarded                       bool   `json:"source_raw_dream_text_forwarded"`
	SourceJanusSurfaceAllowed                         bool   `json:"source_janus_surface_allowed"`
	SourceCoocLearningAllowed                         bool   `json:"source_cooc_learning_allowed"`
	SourceDeltaHarvestAllowed                         bool   `json:"source_delta_harvest_allowed"`
	SourceBodyMutationAllowed                         bool   `json:"source_body_mutation_allowed"`
	SourceRollbackRequired                            bool   `json:"source_rollback_required"`
	SourcePreStateHashRequired                        bool   `json:"source_pre_state_hash_required"`
	SourcePostStateHashRequired                       bool   `json:"source_post_state_hash_required"`
	BodySmokeWeighted                                 bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                  bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                               bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                      bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                           bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                            bool   `json:"source_authority_granted"`
	AuthorityGranted                                  bool   `json:"authority_granted"`
	ContractsReady                                    bool   `json:"contracts_ready"`
	WriteAllowed                                      bool   `json:"write_allowed"`
	AdmissionAllowed                                  bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                              bool   `json:"live_admission_enabled"`
	MutatesState                                      bool   `json:"mutates_state"`
	BodyTarget                                        string `json:"body_target"`
	Passed                                            bool   `json:"passed"`
	Reason                                            string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundary(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-boundary RESONANCE_OBSERVATION_REPORT RESONANCE_GRAFT_BOUNDARY_REPORT")
	}
	observationPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft boundary output path missing")
	}
	observation, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceObservationReportForAssert(observationPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceObservationReportError(observation, root); err != nil {
		return err
	}
	boundary := admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport{
		Schema:     admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema,
		Status:     "shadow_graft_boundary_declared_dry_run",
		Target:     "resonance",
		TargetKind: "weighted_internal_world_shadow_graft",
		TargetMode: "receipt_only_closed_dry_run",
		Action:     "declare_weighted_resonance_shadow_graft_boundary_dry_run",
		WeightedAdmissionResonanceGraftBoundaryReady:  true,
		WeightedAdmissionResonanceObservationConsumed: true,
		WeightedAdmissionResonanceObservationRequired: true,
		NextStepBlockedWithoutResonanceGraftBoundary:  true,
		ReceiptShape:          "weighted_resonance_observation_shadow_graft_boundary",
		BoundaryKind:          "shadow_graft_boundary",
		BoundaryMode:          "no_mutation_receipt",
		BoundaryStage:         "pre_live_graft",
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
		SourceSchema:          observation.Schema,
		SourceStatus:          observation.Status,
		SourceTarget:          observation.Target,
		SourceReport:          observationPath,
		SourceWeightedAdmissionResonanceObservationID:     observation.WeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:  observation.WeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceObservationCausal: observation.CausalID,
		SourceWeightedAdmissionResonanceObservationAppend: observation.AppendHash,
		SourceWeightedAdmissionResonanceObservationRead:   observation.ReadBackHash,
		SourceObserver:                                 observation.Observer,
		SourceObserverKind:                             observation.ObserverKind,
		SourceObservationKind:                          observation.ObservationKind,
		SourceObservationMode:                          observation.ObservationMode,
		SourceAppendOnly:                               observation.AppendOnly,
		SourceReadBack:                                 observation.ReadBack,
		SourceReceiptVerified:                          observation.ReceiptVerified,
		SourceDryRunOnly:                               observation.DryRunOnly,
		SourceObservationRawDreamTextObserved:          observation.RawDreamTextObserved,
		SourceObservationRawDreamTextForwarded:         observation.RawDreamTextForwarded,
		SourceObservationJanusSurfaceAllowed:           observation.JanusSurfaceAllowed,
		SourceObservationCoocLearningAllowed:           observation.CoocLearningAllowed,
		SourceObservationDeltaHarvestAllowed:           observation.DeltaHarvestAllowed,
		SourceObservationBodyMutationAllowed:           observation.BodyMutationAllowed,
		SourceObservationRollbackRequired:              observation.RollbackRequired,
		SourceResonanceReceiverReport:                  observation.SourceReport,
		SourceResonanceIntentReport:                    observation.SourceResonanceIntentReport,
		SourceFinalGateReport:                          observation.SourceFinalGateReport,
		SourceSealReport:                               observation.SourceSealReport,
		SourcePermitReport:                             observation.SourcePermitReport,
		SourceAuthorityReport:                          observation.SourceAuthorityReport,
		SourceContractReport:                           observation.SourceContractReport,
		SourcePreconditionReport:                       observation.SourcePreconditionReport,
		SourceReadinessReport:                          observation.SourceReadinessReport,
		SourceBodyWorkdir:                              observation.SourceBodyWorkdir,
		SourceBoundaryReport:                           observation.SourceBoundaryReport,
		SourceProofLog:                                 observation.SourceProofLog,
		SourceFinalGateLog:                             observation.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceReceiverID:     observation.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:  observation.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceReceiverCausal: observation.SourceWeightedAdmissionResonanceReceiverCausal,
		SourceReceiverPreStateHash:                     observation.SourceReceiverPreStateHash,
		SourceReceiverPostStateHash:                    observation.SourceReceiverPostStateHash,
		SourceReceiverStateDeltaHash:                   observation.SourceReceiverStateDeltaHash,
		SourceWeightedAdmissionResonanceIntentConsumed: observation.SourceWeightedAdmissionResonanceIntentConsumed,
		SourceWeightedAdmissionResonanceIntentRequired: observation.SourceWeightedAdmissionResonanceIntentRequired,
		SourceWeightedAdmissionResonanceIntentReady:    observation.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:       observation.SourceWeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:       observation.SourceWeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:          observation.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:            observation.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:            observation.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:               observation.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:          observation.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:          observation.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:             observation.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:       observation.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:       observation.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                    observation.SourceManualPermitRequested,
		SourcePermitKeyMatched:                         observation.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                      observation.SourceRawDreamTextAllowed,
		SourceRawDreamTextObserved:                     observation.SourceRawDreamTextObserved,
		SourceRawDreamTextForwarded:                    observation.SourceRawDreamTextForwarded,
		SourceJanusSurfaceAllowed:                      observation.SourceJanusSurfaceAllowed,
		SourceCoocLearningAllowed:                      observation.SourceCoocLearningAllowed,
		SourceDeltaHarvestAllowed:                      observation.SourceDeltaHarvestAllowed,
		SourceBodyMutationAllowed:                      observation.SourceBodyMutationAllowed,
		SourceRollbackRequired:                         observation.SourceRollbackRequired,
		SourcePreStateHashRequired:                     observation.SourcePreStateHashRequired,
		SourcePostStateHashRequired:                    observation.SourcePostStateHashRequired,
		BodySmokeWeighted:                              observation.BodySmokeWeighted,
		NanoDirectRunner:                               observation.NanoDirectRunner,
		NanoDirectFinalGate:                            observation.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                   observation.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                        observation.BoundaryReportFullChain,
		SourceAuthorityGranted:                         observation.SourceAuthorityGranted,
		AuthorityGranted:                               false,
		ContractsReady:                                 false,
		WriteAllowed:                                   false,
		AdmissionAllowed:                               false,
		LiveAdmissionEnabled:                           false,
		MutatesState:                                   false,
		BodyTarget:                                     "none",
		Passed:                                         true,
		Reason:                                         "weighted resonance shadow graft boundary declared without body mutation",
	}
	boundary.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryCausalID(boundary)
	boundary.BoundaryHash = admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryHash(boundary)
	boundary.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReadBackHash(boundary)
	boundary.WeightedAdmissionResonanceGraftBoundaryID = admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryID(boundary)
	if boundary.CausalID == "" ||
		boundary.BoundaryHash == "" ||
		boundary.ReadBackHash == "" ||
		boundary.WeightedAdmissionResonanceGraftBoundaryID == "" ||
		boundary.BoundaryHash == boundary.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft boundary read-back proof failed")
	}
	raw, err := json.MarshalIndent(boundary, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft boundary marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft boundary write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-boundary] pass: resonance_graft_boundary_report=%s resonance_observation_report=%s\n", outputPath, observationPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-boundary-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft boundary schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft boundary schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema)
	}
	if report.Status != "shadow_graft_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft boundary status mismatch: got %q want %q", report.Status, "shadow_graft_boundary_declared_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance graft boundary target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft" {
		return fmt.Errorf("weighted admission resonance graft boundary target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft")
	}
	if report.TargetMode != "receipt_only_closed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft boundary target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_dry_run")
	}
	if report.Action != "declare_weighted_resonance_shadow_graft_boundary_dry_run" {
		return fmt.Errorf("weighted admission resonance graft boundary action mismatch: got %q want %q", report.Action, "declare_weighted_resonance_shadow_graft_boundary_dry_run")
	}
	if report.ReceiptShape != "weighted_resonance_observation_shadow_graft_boundary" {
		return fmt.Errorf("weighted admission resonance graft boundary receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_observation_shadow_graft_boundary")
	}
	if report.BoundaryKind != "shadow_graft_boundary" {
		return fmt.Errorf("weighted admission resonance graft boundary boundary_kind mismatch: got %q want %q", report.BoundaryKind, "shadow_graft_boundary")
	}
	if report.BoundaryMode != "no_mutation_receipt" {
		return fmt.Errorf("weighted admission resonance graft boundary boundary_mode mismatch: got %q want %q", report.BoundaryMode, "no_mutation_receipt")
	}
	if report.BoundaryStage != "pre_live_graft" {
		return fmt.Errorf("weighted admission resonance graft boundary boundary_stage mismatch: got %q want %q", report.BoundaryStage, "pre_live_graft")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_boundary_ready", report.WeightedAdmissionResonanceGraftBoundaryReady},
		{"weighted_admission_resonance_observation_consumed", report.WeightedAdmissionResonanceObservationConsumed},
		{"weighted_admission_resonance_observation_required", report.WeightedAdmissionResonanceObservationRequired},
		{"next_step_blocked_without_resonance_graft_boundary", report.NextStepBlockedWithoutResonanceGraftBoundary},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
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
			return fmt.Errorf("weighted admission resonance graft boundary %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft boundary opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
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
			return fmt.Errorf("weighted admission resonance graft boundary %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema {
		return fmt.Errorf("weighted admission resonance graft boundary source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceObservationSchema)
	}
	if report.SourceStatus != "observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft boundary source_status mismatch: got %q want %q", report.SourceStatus, "observation_recorded_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft boundary source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft boundary source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationCausal, "weighted-resonance-observation-causal-") {
		return fmt.Errorf("weighted admission resonance graft boundary source observation causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationAppend, "weighted-resonance-observation-append-") {
		return fmt.Errorf("weighted admission resonance graft boundary source observation append prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationRead, "weighted-resonance-observation-read-") {
		return fmt.Errorf("weighted admission resonance graft boundary source observation read-back prefix mismatch")
	}
	if report.SourceObserver != "resonance" {
		return fmt.Errorf("weighted admission resonance graft boundary source_observer mismatch: got %q want %q", report.SourceObserver, "resonance")
	}
	if report.SourceObserverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance graft boundary source_observer_kind mismatch: got %q want %q", report.SourceObserverKind, "internal_world")
	}
	if report.SourceObservationKind != "weighted_receiver_state_proof" {
		return fmt.Errorf("weighted admission resonance graft boundary source_observation_kind mismatch: got %q want %q", report.SourceObservationKind, "weighted_receiver_state_proof")
	}
	if report.SourceObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft boundary source_observation_mode mismatch: got %q want %q", report.SourceObservationMode, "sealed_metadata_observation")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft boundary source receiver id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") {
		return fmt.Errorf("weighted admission resonance graft boundary source receiver causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(report.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(report.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		report.SourceReceiverPreStateHash == report.SourceReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft boundary source receiver state proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft boundary body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft boundary causal_id mismatch")
	}
	if report.BoundaryHash == "" || report.BoundaryHash != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryHash(report) {
		return fmt.Errorf("weighted admission resonance graft boundary boundary_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft boundary read_back_hash mismatch")
	}
	if report.BoundaryHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft boundary read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftBoundaryID == "" || report.WeightedAdmissionResonanceGraftBoundaryID != admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryID(report) {
		return fmt.Errorf("weighted admission resonance graft boundary id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft boundary declared without body mutation" {
		return fmt.Errorf("weighted admission resonance graft boundary reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryCausalID(boundary admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport) string {
	h := hashJSON(struct {
		SourceObservationID           string `json:"source_observation_id"`
		SourceObservationReadBackHash string `json:"source_observation_read_back_hash"`
		SourceReceiverID              string `json:"source_receiver_id"`
		Target                        string `json:"target"`
		BoundaryKind                  string `json:"boundary_kind"`
		BoundaryStage                 string `json:"boundary_stage"`
	}{
		SourceObservationID:           boundary.SourceWeightedAdmissionResonanceObservationID,
		SourceObservationReadBackHash: boundary.SourceWeightedAdmissionResonanceObservationRead,
		SourceReceiverID:              boundary.SourceWeightedAdmissionResonanceReceiverID,
		Target:                        boundary.Target,
		BoundaryKind:                  boundary.BoundaryKind,
		BoundaryStage:                 boundary.BoundaryStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-boundary-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryHash(boundary admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport) string {
	h := hashJSON(struct {
		CausalID                  string `json:"causal_id"`
		SourceObservationID       string `json:"source_observation_id"`
		SourceObservationAppend   string `json:"source_observation_append_hash"`
		SourceObservationReadBack string `json:"source_observation_read_back_hash"`
		BoundaryMode              string `json:"boundary_mode"`
		ShadowOnly                bool   `json:"shadow_only"`
		DryRunOnly                bool   `json:"dry_run_only"`
		GraftAllowed              bool   `json:"graft_allowed"`
	}{
		CausalID:                  boundary.CausalID,
		SourceObservationID:       boundary.SourceWeightedAdmissionResonanceObservationID,
		SourceObservationAppend:   boundary.SourceWeightedAdmissionResonanceObservationAppend,
		SourceObservationReadBack: boundary.SourceWeightedAdmissionResonanceObservationRead,
		BoundaryMode:              boundary.BoundaryMode,
		ShadowOnly:                boundary.ShadowOnly,
		DryRunOnly:                boundary.DryRunOnly,
		GraftAllowed:              boundary.GraftAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-boundary-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReadBackHash(boundary admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport) string {
	h := hashJSON(struct {
		BoundaryHash    string `json:"boundary_hash"`
		ObservationID   string `json:"observation_id"`
		BoundaryKind    string `json:"boundary_kind"`
		BoundaryReady   bool   `json:"boundary_ready"`
		BodyMutation    bool   `json:"body_mutation"`
		AdmissionOpened bool   `json:"admission_opened"`
	}{
		BoundaryHash:    boundary.BoundaryHash,
		ObservationID:   boundary.SourceWeightedAdmissionResonanceObservationID,
		BoundaryKind:    boundary.BoundaryKind,
		BoundaryReady:   boundary.WeightedAdmissionResonanceGraftBoundaryReady,
		BodyMutation:    boundary.BodyMutationAllowed,
		AdmissionOpened: boundary.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-boundary-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryID(boundary admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		BoundaryHash            string `json:"boundary_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		BoundaryKind            string `json:"boundary_kind"`
		BoundaryMode            string `json:"boundary_mode"`
		BoundaryStage           string `json:"boundary_stage"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_boundary"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
	}{
		Schema:                  boundary.Schema,
		Status:                  boundary.Status,
		Action:                  boundary.Action,
		SourceReport:            boundary.SourceReport,
		SourceObservationID:     boundary.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        boundary.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                boundary.CausalID,
		BoundaryHash:            boundary.BoundaryHash,
		ReadBackHash:            boundary.ReadBackHash,
		Ready:                   boundary.WeightedAdmissionResonanceGraftBoundaryReady,
		ReceiptShape:            boundary.ReceiptShape,
		BoundaryKind:            boundary.BoundaryKind,
		BoundaryMode:            boundary.BoundaryMode,
		BoundaryStage:           boundary.BoundaryStage,
		ShadowOnly:              boundary.ShadowOnly,
		GraftAllowed:            boundary.GraftAllowed,
		DryRunOnly:              boundary.DryRunOnly,
		RawDreamTextAllowed:     boundary.RawDreamTextAllowed,
		JanusSurfaceAllowed:     boundary.JanusSurfaceAllowed,
		CoocLearningAllowed:     boundary.CoocLearningAllowed,
		DeltaHarvestAllowed:     boundary.DeltaHarvestAllowed,
		BodyMutationAllowed:     boundary.BodyMutationAllowed,
		RollbackRequired:        boundary.RollbackRequired,
		LiveReady:               boundary.LiveReady,
		ContractsReady:          boundary.ContractsReady,
		BodyTarget:              boundary.BodyTarget,
		WriteAllowed:            boundary.WriteAllowed,
		AdmissionAllowed:        boundary.AdmissionAllowed,
		LiveAdmissionEnabled:    boundary.LiveAdmissionEnabled,
		MutatesState:            boundary.MutatesState,
		NextStepBlockedWithout:  boundary.NextStepBlockedWithoutResonanceGraftBoundary,
		SourceObservationReady:  boundary.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     boundary.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       boundary.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    boundary.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         boundary.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       boundary.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: boundary.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-boundary-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftBoundaryReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft boundary path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft boundary not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft boundary not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft boundary JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft boundary decode failed: %w", err)
	}
	return report, root, nil
}
