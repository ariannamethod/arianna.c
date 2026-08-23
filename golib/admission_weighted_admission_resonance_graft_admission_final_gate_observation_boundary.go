package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport

	AdmissionFinalGateObservationBoundaryState                                string `json:"admission_final_gate_observation_boundary_state"`
	AdmissionFinalGateObservationBoundaryAction                               string `json:"admission_final_gate_observation_boundary_action"`
	AdmissionFinalGateObservationBoundaryTarget                               string `json:"admission_final_gate_observation_boundary_target"`
	AdmissionFinalGateObservationBoundaryTargetKind                           string `json:"admission_final_gate_observation_boundary_target_kind"`
	AdmissionFinalGateObservationBoundaryTargetMode                           string `json:"admission_final_gate_observation_boundary_target_mode"`
	AdmissionFinalGateObservationBoundaryDryRunOnly                           bool   `json:"admission_final_gate_observation_boundary_dry_run_only"`
	AdmissionFinalGateObservationBoundaryObservationVerified                  bool   `json:"admission_final_gate_observation_boundary_observation_verified"`
	AdmissionFinalGateObservationBoundaryReadBackVerified                     bool   `json:"admission_final_gate_observation_boundary_read_back_verified"`
	AdmissionFinalGateObservationBoundaryReady                                bool   `json:"admission_final_gate_observation_boundary_ready"`
	FinalGateObservationBoundaryKind                                          string `json:"final_gate_observation_boundary_kind"`
	FinalGateObservationBoundaryMode                                          string `json:"final_gate_observation_boundary_mode"`
	FinalGateObservationBoundaryStage                                         string `json:"final_gate_observation_boundary_stage"`
	FinalGateObservationBoundaryRawDreamTextObserved                          bool   `json:"final_gate_observation_boundary_raw_dream_text_observed"`
	FinalGateObservationBoundaryRawDreamTextForwarded                         bool   `json:"final_gate_observation_boundary_raw_dream_text_forwarded"`
	FinalGateObservationBoundaryRawDreamTextAllowed                           bool   `json:"final_gate_observation_boundary_raw_dream_text_allowed"`
	FinalGateObservationBoundaryJanusSurfaceAllowed                           bool   `json:"final_gate_observation_boundary_janus_surface_allowed"`
	FinalGateObservationBoundaryCoocLearningAllowed                           bool   `json:"final_gate_observation_boundary_cooc_learning_allowed"`
	FinalGateObservationBoundaryDeltaHarvestAllowed                           bool   `json:"final_gate_observation_boundary_delta_harvest_allowed"`
	FinalGateObservationBoundaryBodyMutationAllowed                           bool   `json:"final_gate_observation_boundary_body_mutation_allowed"`
	FinalGateObservationBoundaryPreStateHashRequired                          bool   `json:"final_gate_observation_boundary_pre_state_hash_required"`
	FinalGateObservationBoundaryPostStateHashRequired                         bool   `json:"final_gate_observation_boundary_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationConsumed      bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationRequired      bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID    string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id"`
	AdmissionFinalGateObservationBoundaryHash                                 string `json:"admission_final_gate_observation_boundary_hash"`
	AdmissionFinalGateObservationBoundaryReadBackHash                         string `json:"admission_final_gate_observation_boundary_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID      string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady   bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal  string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_causal_id"`
	SourceAdmissionFinalGateObservationAppendHash                             string `json:"source_admission_final_gate_observation_append_hash"`
	SourceAdmissionFinalGateObservationReadBackHash                           string `json:"source_admission_final_gate_observation_read_back_hash"`
	SourceAdmissionFinalGateObservationReceiptShape                           string `json:"source_admission_final_gate_observation_receipt_shape"`
	SourceAdmissionFinalGateObservationState                                  string `json:"source_admission_final_gate_observation_state"`
	SourceAdmissionFinalGateObservationAction                                 string `json:"source_admission_final_gate_observation_action"`
	SourceAdmissionFinalGateObservationTarget                                 string `json:"source_admission_final_gate_observation_target"`
	SourceAdmissionFinalGateObservationTargetKind                             string `json:"source_admission_final_gate_observation_target_kind"`
	SourceAdmissionFinalGateObservationTargetMode                             string `json:"source_admission_final_gate_observation_target_mode"`
	SourceAdmissionFinalGateObservationDryRunOnly                             bool   `json:"source_admission_final_gate_observation_dry_run_only"`
	SourceAdmissionFinalGateObservationAppendOnly                             bool   `json:"source_admission_final_gate_observation_append_only"`
	SourceAdmissionFinalGateObservationReadBack                               bool   `json:"source_admission_final_gate_observation_read_back"`
	SourceAdmissionFinalGateObservationReceiptVerified                        bool   `json:"source_admission_final_gate_observation_receipt_verified"`
	SourceAdmissionFinalGateObservationReceiverVerified                       bool   `json:"source_admission_final_gate_observation_receiver_verified"`
	SourceAdmissionFinalGateObservationReady                                  bool   `json:"source_admission_final_gate_observation_ready"`
	SourceFinalGateObservationObserver                                        string `json:"source_final_gate_observation_observer"`
	SourceFinalGateObservationObserverKind                                    string `json:"source_final_gate_observation_observer_kind"`
	SourceFinalGateObservationKind                                            string `json:"source_final_gate_observation_kind"`
	SourceFinalGateObservationMode                                            string `json:"source_final_gate_observation_mode"`
	SourceFinalGateObservationRawDreamTextObserved                            bool   `json:"source_final_gate_observation_raw_dream_text_observed"`
	SourceFinalGateObservationRawDreamTextForwarded                           bool   `json:"source_final_gate_observation_raw_dream_text_forwarded"`
	SourceFinalGateObservationRawDreamTextAllowed                             bool   `json:"source_final_gate_observation_raw_dream_text_allowed"`
	SourceFinalGateObservationJanusSurfaceAllowed                             bool   `json:"source_final_gate_observation_janus_surface_allowed"`
	SourceFinalGateObservationCoocLearningAllowed                             bool   `json:"source_final_gate_observation_cooc_learning_allowed"`
	SourceFinalGateObservationDeltaHarvestAllowed                             bool   `json:"source_final_gate_observation_delta_harvest_allowed"`
	SourceFinalGateObservationBodyMutationAllowed                             bool   `json:"source_final_gate_observation_body_mutation_allowed"`
	SourceFinalGateObservationPreStateHashRequired                            bool   `json:"source_final_gate_observation_pre_state_hash_required"`
	SourceFinalGateObservationPostStateHashRequired                           bool   `json:"source_final_gate_observation_post_state_hash_required"`
	SourceAdmissionFinalGateObservationReason                                 string `json:"source_admission_final_gate_observation_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT")
	}
	observationPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary output path missing")
	}
	sourceObservation, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportForAssert(observationPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportError(sourceObservation, root); err != nil {
		return err
	}
	boundary := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport: sourceObservation,
		AdmissionFinalGateObservationBoundaryState:                                           "declared",
		AdmissionFinalGateObservationBoundaryAction:                                          "declare_blocked_final_gate_observation_boundary",
		AdmissionFinalGateObservationBoundaryTarget:                                          "resonance",
		AdmissionFinalGateObservationBoundaryTargetKind:                                      "weighted_internal_world_shadow_graft_admission_final_gate_observation",
		AdmissionFinalGateObservationBoundaryTargetMode:                                      "receipt_only_closed_dry_run",
		AdmissionFinalGateObservationBoundaryDryRunOnly:                                      true,
		AdmissionFinalGateObservationBoundaryObservationVerified:                             true,
		AdmissionFinalGateObservationBoundaryReadBackVerified:                                true,
		AdmissionFinalGateObservationBoundaryReady:                                           false,
		FinalGateObservationBoundaryKind:                                                     "blocked_final_gate_observation_boundary",
		FinalGateObservationBoundaryMode:                                                     "no_mutation_closed_boundary_receipt",
		FinalGateObservationBoundaryStage:                                                    "post_observation_pre_live_admission",
		FinalGateObservationBoundaryRawDreamTextObserved:                                     false,
		FinalGateObservationBoundaryRawDreamTextForwarded:                                    false,
		FinalGateObservationBoundaryRawDreamTextAllowed:                                      false,
		FinalGateObservationBoundaryJanusSurfaceAllowed:                                      false,
		FinalGateObservationBoundaryCoocLearningAllowed:                                      false,
		FinalGateObservationBoundaryDeltaHarvestAllowed:                                      false,
		FinalGateObservationBoundaryBodyMutationAllowed:                                      false,
		FinalGateObservationBoundaryPreStateHashRequired:                                     true,
		FinalGateObservationBoundaryPostStateHashRequired:                                    true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady:            true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationConsumed:                 true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationRequired:                 true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary:            true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID:                 sourceObservation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:              sourceObservation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal:             sourceObservation.CausalID,
		SourceAdmissionFinalGateObservationAppendHash:                                        sourceObservation.AdmissionFinalGateObservationAppendHash,
		SourceAdmissionFinalGateObservationReadBackHash:                                      sourceObservation.AdmissionFinalGateObservationReadBackHash,
		SourceAdmissionFinalGateObservationReceiptShape:                                      sourceObservation.ReceiptShape,
		SourceAdmissionFinalGateObservationState:                                             sourceObservation.AdmissionFinalGateObservationState,
		SourceAdmissionFinalGateObservationAction:                                            sourceObservation.AdmissionFinalGateObservationAction,
		SourceAdmissionFinalGateObservationTarget:                                            sourceObservation.AdmissionFinalGateObservationTarget,
		SourceAdmissionFinalGateObservationTargetKind:                                        sourceObservation.AdmissionFinalGateObservationTargetKind,
		SourceAdmissionFinalGateObservationTargetMode:                                        sourceObservation.AdmissionFinalGateObservationTargetMode,
		SourceAdmissionFinalGateObservationDryRunOnly:                                        sourceObservation.AdmissionFinalGateObservationDryRunOnly,
		SourceAdmissionFinalGateObservationAppendOnly:                                        sourceObservation.AdmissionFinalGateObservationAppendOnly,
		SourceAdmissionFinalGateObservationReadBack:                                          sourceObservation.AdmissionFinalGateObservationReadBack,
		SourceAdmissionFinalGateObservationReceiptVerified:                                   sourceObservation.AdmissionFinalGateObservationReceiptVerified,
		SourceAdmissionFinalGateObservationReceiverVerified:                                  sourceObservation.AdmissionFinalGateObservationReceiverVerified,
		SourceAdmissionFinalGateObservationReady:                                             sourceObservation.AdmissionFinalGateObservationReady,
		SourceFinalGateObservationObserver:                                                   sourceObservation.FinalGateObservationObserver,
		SourceFinalGateObservationObserverKind:                                               sourceObservation.FinalGateObservationObserverKind,
		SourceFinalGateObservationKind:                                                       sourceObservation.FinalGateObservationKind,
		SourceFinalGateObservationMode:                                                       sourceObservation.FinalGateObservationMode,
		SourceFinalGateObservationRawDreamTextObserved:                                       sourceObservation.FinalGateObservationRawDreamTextObserved,
		SourceFinalGateObservationRawDreamTextForwarded:                                      sourceObservation.FinalGateObservationRawDreamTextForwarded,
		SourceFinalGateObservationRawDreamTextAllowed:                                        sourceObservation.FinalGateObservationRawDreamTextAllowed,
		SourceFinalGateObservationJanusSurfaceAllowed:                                        sourceObservation.FinalGateObservationJanusSurfaceAllowed,
		SourceFinalGateObservationCoocLearningAllowed:                                        sourceObservation.FinalGateObservationCoocLearningAllowed,
		SourceFinalGateObservationDeltaHarvestAllowed:                                        sourceObservation.FinalGateObservationDeltaHarvestAllowed,
		SourceFinalGateObservationBodyMutationAllowed:                                        sourceObservation.FinalGateObservationBodyMutationAllowed,
		SourceFinalGateObservationPreStateHashRequired:                                       sourceObservation.FinalGateObservationPreStateHashRequired,
		SourceFinalGateObservationPostStateHashRequired:                                      sourceObservation.FinalGateObservationPostStateHashRequired,
		SourceAdmissionFinalGateObservationReason:                                            sourceObservation.Reason,
	}
	boundary.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema
	boundary.Status = "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run"
	boundary.Target = "live_route_admission_next_step"
	boundary.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary"
	boundary.TargetMode = "receipt_only_closed_dry_run"
	boundary.Action = "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run"
	boundary.WriterAction = "reject_blocked_admission_final_gate_observation_boundary"
	boundary.RollbackAction = "reject_blocked_admission_final_gate_observation_boundary"
	boundary.LedgerState = "blocked"
	boundary.LedgerAction = "reject_blocked_admission_final_gate_observation_boundary"
	boundary.LedgerContract = "none"
	boundary.LedgerEntrypoint = "none"
	boundary.LedgerReceiptShape = "none"
	boundary.LedgerWriteScope = "none"
	boundary.LedgerReady = false
	boundary.LedgerAppendAllowed = false
	boundary.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_receipt"
	boundary.SourceSchema = sourceObservation.Schema
	boundary.SourceStatus = sourceObservation.Status
	boundary.SourceTarget = sourceObservation.Target
	boundary.SourceReport = observationPath
	boundary.AuthorityGranted = false
	boundary.Reason = "weighted resonance shadow graft admission final gate observation boundary declared from recorded observation; live admission remains closed"
	boundary.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryCausalID(boundary)
	boundary.AdmissionFinalGateObservationBoundaryHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryHash(boundary)
	boundary.AdmissionFinalGateObservationBoundaryReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReadBackHash(boundary)
	boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID(boundary)
	if boundary.CausalID == "" ||
		boundary.AdmissionFinalGateObservationBoundaryHash == "" ||
		boundary.AdmissionFinalGateObservationBoundaryReadBackHash == "" ||
		boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID == "" ||
		boundary.AdmissionFinalGateObservationBoundaryHash == boundary.AdmissionFinalGateObservationBoundaryReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary read-back proof failed")
	}
	raw, err := json.MarshalIndent(boundary, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary] pass: resonance_graft_admission_final_gate_observation_boundary_report=%s resonance_graft_admission_final_gate_observation_report=%s\n", outputPath, observationPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary")
	}
	if report.TargetMode != "receipt_only_closed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary target_mode mismatch: got %q want %q", report.TargetMode, "receipt_only_closed_dry_run")
	}
	if report.Action != "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary action mismatch: got %q want %q", report.Action, "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_observation_boundary" || report.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary ledger guard mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryState != "declared" ||
		report.AdmissionFinalGateObservationBoundaryAction != "declare_blocked_final_gate_observation_boundary" ||
		report.AdmissionFinalGateObservationBoundaryTarget != "resonance" ||
		report.AdmissionFinalGateObservationBoundaryTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation" ||
		report.AdmissionFinalGateObservationBoundaryTargetMode != "receipt_only_closed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_dry_run_only", report.AdmissionFinalGateObservationBoundaryDryRunOnly},
		{"admission_final_gate_observation_boundary_observation_verified", report.AdmissionFinalGateObservationBoundaryObservationVerified},
		{"admission_final_gate_observation_boundary_read_back_verified", report.AdmissionFinalGateObservationBoundaryReadBackVerified},
		{"final_gate_observation_boundary_pre_state_hash_required", report.FinalGateObservationBoundaryPreStateHashRequired},
		{"final_gate_observation_boundary_post_state_hash_required", report.FinalGateObservationBoundaryPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady},
		{"source_admission_final_gate_observation_dry_run_only", report.SourceAdmissionFinalGateObservationDryRunOnly},
		{"source_admission_final_gate_observation_append_only", report.SourceAdmissionFinalGateObservationAppendOnly},
		{"source_admission_final_gate_observation_read_back", report.SourceAdmissionFinalGateObservationReadBack},
		{"source_admission_final_gate_observation_receipt_verified", report.SourceAdmissionFinalGateObservationReceiptVerified},
		{"source_final_gate_observation_pre_state_hash_required", report.SourceFinalGateObservationPreStateHashRequired},
		{"source_final_gate_observation_post_state_hash_required", report.SourceFinalGateObservationPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady},
		{"weighted_admission_resonance_graft_admission_final_gate_intent_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_intent_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady},
		{"weighted_admission_resonance_graft_admission_final_gate_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady},
		{"weighted_admission_resonance_graft_admission_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionSealReady},
		{"weighted_admission_resonance_graft_admission_authority_ready", report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady},
		{"weighted_admission_resonance_graft_admission_permit_ready", report.WeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"weighted_admission_resonance_graft_admission_readiness_ready", report.WeightedAdmissionResonanceGraftAdmissionReadinessReady},
		{"writer_inventory_verified", report.WriterInventoryVerified},
		{"writer_preflight_verified", report.WriterPreflightVerified},
		{"live_ready", report.LiveReady},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_ready", report.AdmissionFinalGateObservationBoundaryReady},
		{"final_gate_observation_boundary_raw_dream_text_observed", report.FinalGateObservationBoundaryRawDreamTextObserved},
		{"final_gate_observation_boundary_raw_dream_text_forwarded", report.FinalGateObservationBoundaryRawDreamTextForwarded},
		{"final_gate_observation_boundary_raw_dream_text_allowed", report.FinalGateObservationBoundaryRawDreamTextAllowed},
		{"final_gate_observation_boundary_janus_surface_allowed", report.FinalGateObservationBoundaryJanusSurfaceAllowed},
		{"final_gate_observation_boundary_cooc_learning_allowed", report.FinalGateObservationBoundaryCoocLearningAllowed},
		{"final_gate_observation_boundary_delta_harvest_allowed", report.FinalGateObservationBoundaryDeltaHarvestAllowed},
		{"final_gate_observation_boundary_body_mutation_allowed", report.FinalGateObservationBoundaryBodyMutationAllowed},
		{"source_admission_final_gate_observation_receiver_verified", report.SourceAdmissionFinalGateObservationReceiverVerified},
		{"source_admission_final_gate_observation_ready", report.SourceAdmissionFinalGateObservationReady},
		{"source_final_gate_observation_raw_dream_text_observed", report.SourceFinalGateObservationRawDreamTextObserved},
		{"source_final_gate_observation_raw_dream_text_forwarded", report.SourceFinalGateObservationRawDreamTextForwarded},
		{"source_final_gate_observation_raw_dream_text_allowed", report.SourceFinalGateObservationRawDreamTextAllowed},
		{"source_final_gate_observation_janus_surface_allowed", report.SourceFinalGateObservationJanusSurfaceAllowed},
		{"source_final_gate_observation_cooc_learning_allowed", report.SourceFinalGateObservationCoocLearningAllowed},
		{"source_final_gate_observation_delta_harvest_allowed", report.SourceFinalGateObservationDeltaHarvestAllowed},
		{"source_final_gate_observation_body_mutation_allowed", report.SourceFinalGateObservationBodyMutationAllowed},
		{"admission_final_gate_observation_receiver_verified", report.AdmissionFinalGateObservationReceiverVerified},
		{"admission_final_gate_observation_ready", report.AdmissionFinalGateObservationReady},
		{"final_gate_observation_raw_dream_text_observed", report.FinalGateObservationRawDreamTextObserved},
		{"final_gate_observation_raw_dream_text_forwarded", report.FinalGateObservationRawDreamTextForwarded},
		{"final_gate_observation_raw_dream_text_allowed", report.FinalGateObservationRawDreamTextAllowed},
		{"final_gate_observation_janus_surface_allowed", report.FinalGateObservationJanusSurfaceAllowed},
		{"final_gate_observation_cooc_learning_allowed", report.FinalGateObservationCoocLearningAllowed},
		{"final_gate_observation_delta_harvest_allowed", report.FinalGateObservationDeltaHarvestAllowed},
		{"final_gate_observation_body_mutation_allowed", report.FinalGateObservationBodyMutationAllowed},
		{"admission_final_gate_receiver_ready", report.AdmissionFinalGateReceiverReady},
		{"final_gate_receiver_raw_dream_text_observed", report.FinalGateReceiverRawDreamTextObserved},
		{"final_gate_receiver_raw_dream_text_forwarded", report.FinalGateReceiverRawDreamTextForwarded},
		{"final_gate_receiver_raw_dream_text_allowed", report.FinalGateReceiverRawDreamTextAllowed},
		{"final_gate_receiver_janus_surface_allowed", report.FinalGateReceiverJanusSurfaceAllowed},
		{"final_gate_receiver_cooc_learning_allowed", report.FinalGateReceiverCoocLearningAllowed},
		{"final_gate_receiver_delta_harvest_allowed", report.FinalGateReceiverDeltaHarvestAllowed},
		{"final_gate_receiver_body_mutation_allowed", report.FinalGateReceiverBodyMutationAllowed},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"final_gate_intent_raw_dream_text_allowed", report.FinalGateIntentRawDreamTextAllowed},
		{"final_gate_intent_janus_surface_allowed", report.FinalGateIntentJanusSurfaceAllowed},
		{"final_gate_intent_cooc_learning_allowed", report.FinalGateIntentCoocLearningAllowed},
		{"final_gate_intent_delta_harvest_allowed", report.FinalGateIntentDeltaHarvestAllowed},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary opened %s", closed.name)
		}
	}
	if report.FinalGateObservationBoundaryKind != "blocked_final_gate_observation_boundary" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary boundary_kind mismatch: got %q want %q", report.FinalGateObservationBoundaryKind, "blocked_final_gate_observation_boundary")
	}
	if report.FinalGateObservationBoundaryMode != "no_mutation_closed_boundary_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary boundary_mode mismatch: got %q want %q", report.FinalGateObservationBoundaryMode, "no_mutation_closed_boundary_receipt")
	}
	if report.FinalGateObservationBoundaryStage != "post_observation_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary boundary_stage mismatch: got %q want %q", report.FinalGateObservationBoundaryStage, "post_observation_pre_live_admission")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_observation_boundary_hash", report.AdmissionFinalGateObservationBoundaryHash},
		{"admission_final_gate_observation_boundary_read_back_hash", report.AdmissionFinalGateObservationBoundaryReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal},
		{"source_admission_final_gate_observation_append_hash", report.SourceAdmissionFinalGateObservationAppendHash},
		{"source_admission_final_gate_observation_read_back_hash", report.SourceAdmissionFinalGateObservationReadBackHash},
		{"source_admission_final_gate_observation_reason", report.SourceAdmissionFinalGateObservationReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_intent_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID},
		{"source_weighted_admission_resonance_graft_admission_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSealID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_recorded_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateObservationReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt" ||
		report.SourceAdmissionFinalGateObservationState != "recorded" ||
		report.SourceAdmissionFinalGateObservationAction != "record_blocked_final_gate_receiver_observation" ||
		report.SourceAdmissionFinalGateObservationTarget != "resonance" ||
		report.SourceAdmissionFinalGateObservationTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_receiver" ||
		report.SourceAdmissionFinalGateObservationTargetMode != "append_only_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source admission final gate observation shape mismatch")
	}
	if report.SourceFinalGateObservationObserver != "resonance" ||
		report.SourceFinalGateObservationObserverKind != "internal_world" ||
		report.SourceFinalGateObservationKind != "blocked_final_gate_receiver_state_proof" ||
		report.SourceFinalGateObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source final gate observation mismatch")
	}
	if report.SourceAdmissionFinalGateObservationReason != "weighted resonance shadow graft admission final gate observation recorded from blocked receiver; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source_admission_final_gate_observation_reason mismatch: got %q", report.SourceAdmissionFinalGateObservationReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID, "weighted-resonance-graft-admission-final-gate-observation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal, "weighted-resonance-graft-admission-final-gate-observation-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationAppendHash, "weighted-resonance-graft-admission-final-gate-observation-append-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-read-") ||
		report.SourceAdmissionFinalGateObservationAppendHash == report.SourceAdmissionFinalGateObservationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary source observation proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary causal_id mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryHash == "" || report.AdmissionFinalGateObservationBoundaryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary boundary_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryReadBackHash == "" || report.AdmissionFinalGateObservationBoundaryReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary read_back_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryHash == report.AdmissionFinalGateObservationBoundaryReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary declared from recorded observation; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport) string {
	h := hashJSON(struct {
		SourceObservationID     string `json:"source_admission_final_gate_observation_id"`
		SourceObservationCausal string `json:"source_admission_final_gate_observation_causal_id"`
		SourceObservationRead   string `json:"source_admission_final_gate_observation_read_back_hash"`
		SourceReport            string `json:"source_report"`
		BoundaryKind            string `json:"boundary_kind"`
		BoundaryStage           string `json:"boundary_stage"`
	}{
		SourceObservationID:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceObservationCausal: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal,
		SourceObservationRead:   report.SourceAdmissionFinalGateObservationReadBackHash,
		SourceReport:            report.SourceReport,
		BoundaryKind:            report.FinalGateObservationBoundaryKind,
		BoundaryStage:           report.FinalGateObservationBoundaryStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport) string {
	h := hashJSON(struct {
		CausalID                string `json:"causal_id"`
		SourceObservationID     string `json:"source_admission_final_gate_observation_id"`
		SourceObservationAppend string `json:"source_admission_final_gate_observation_append_hash"`
		SourceObservationRead   string `json:"source_admission_final_gate_observation_read_back_hash"`
		BoundaryMode            string `json:"boundary_mode"`
		DryRunOnly              bool   `json:"dry_run_only"`
		ObservationVerified     bool   `json:"observation_verified"`
		RawDreamTextVisible     bool   `json:"raw_dream_text_visible"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
	}{
		CausalID:                report.CausalID,
		SourceObservationID:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		SourceObservationAppend: report.SourceAdmissionFinalGateObservationAppendHash,
		SourceObservationRead:   report.SourceAdmissionFinalGateObservationReadBackHash,
		BoundaryMode:            report.FinalGateObservationBoundaryMode,
		DryRunOnly:              report.AdmissionFinalGateObservationBoundaryDryRunOnly,
		ObservationVerified:     report.AdmissionFinalGateObservationBoundaryObservationVerified,
		RawDreamTextVisible:     report.FinalGateObservationBoundaryRawDreamTextObserved || report.FinalGateObservationBoundaryRawDreamTextForwarded,
		BodyMutationAllowed:     report.FinalGateObservationBoundaryBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport) string {
	h := hashJSON(struct {
		BoundaryHash        string `json:"boundary_hash"`
		SourceObservationID string `json:"source_admission_final_gate_observation_id"`
		BoundaryKind        string `json:"boundary_kind"`
		ReadBackVerified    bool   `json:"read_back_verified"`
		BoundaryReady       bool   `json:"boundary_ready"`
		AdmissionOpened     bool   `json:"admission_opened"`
	}{
		BoundaryHash:        report.AdmissionFinalGateObservationBoundaryHash,
		SourceObservationID: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		BoundaryKind:        report.FinalGateObservationBoundaryKind,
		ReadBackVerified:    report.AdmissionFinalGateObservationBoundaryReadBackVerified,
		BoundaryReady:       report.AdmissionFinalGateObservationBoundaryReady,
		AdmissionOpened:     report.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceObservationID     string `json:"source_admission_final_gate_observation_id"`
		CausalID                string `json:"causal_id"`
		BoundaryHash            string `json:"boundary_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"weighted_boundary_ready"`
		BoundaryReady           bool   `json:"admission_final_gate_observation_boundary_ready"`
		ObservationVerified     bool   `json:"observation_verified"`
		ReadBackVerified        bool   `json:"read_back_verified"`
		DryRunOnly              bool   `json:"dry_run_only"`
		RawDreamTextObserved    bool   `json:"raw_dream_text_observed"`
		RawDreamTextForwarded   bool   `json:"raw_dream_text_forwarded"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary"`
		SourceObservationReady  bool   `json:"source_weighted_observation_ready"`
		SourceReceiverConsumed  bool   `json:"source_weighted_receiver_consumed"`
		SourceFinalGateReady    bool   `json:"source_weighted_final_gate_ready"`
		SourceAuthorityGranted  bool   `json:"source_authority_granted"`
		SourceObservationClosed bool   `json:"source_observation_closed"`
	}{
		Schema:                  report.Schema,
		Status:                  report.Status,
		Action:                  report.Action,
		SourceObservationID:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID,
		CausalID:                report.CausalID,
		BoundaryHash:            report.AdmissionFinalGateObservationBoundaryHash,
		ReadBackHash:            report.AdmissionFinalGateObservationBoundaryReadBackHash,
		Ready:                   report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady,
		BoundaryReady:           report.AdmissionFinalGateObservationBoundaryReady,
		ObservationVerified:     report.AdmissionFinalGateObservationBoundaryObservationVerified,
		ReadBackVerified:        report.AdmissionFinalGateObservationBoundaryReadBackVerified,
		DryRunOnly:              report.AdmissionFinalGateObservationBoundaryDryRunOnly,
		RawDreamTextObserved:    report.FinalGateObservationBoundaryRawDreamTextObserved,
		RawDreamTextForwarded:   report.FinalGateObservationBoundaryRawDreamTextForwarded,
		BodyMutationAllowed:     report.FinalGateObservationBoundaryBodyMutationAllowed,
		WriteAllowed:            report.WriteAllowed,
		AdmissionAllowed:        report.AdmissionAllowed,
		LiveAdmissionEnabled:    report.LiveAdmissionEnabled,
		MutatesState:            report.MutatesState,
		NextStepBlockedWithout:  report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary,
		SourceObservationReady:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		SourceReceiverConsumed:  report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed,
		SourceFinalGateReady:    report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceAuthorityGranted:  report.AuthorityGranted,
		SourceObservationClosed: !report.SourceAdmissionFinalGateObservationReady && !report.SourceFinalGateObservationBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary decode failed: %w", err)
	}
	return report, root, nil
}
