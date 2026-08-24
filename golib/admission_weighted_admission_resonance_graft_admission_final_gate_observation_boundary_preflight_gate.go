package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReport

	AdmissionFinalGateObservationBoundaryPreflightGateState                                   string `json:"admission_final_gate_observation_boundary_preflight_gate_state"`
	AdmissionFinalGateObservationBoundaryPreflightGateAction                                  string `json:"admission_final_gate_observation_boundary_preflight_gate_action"`
	AdmissionFinalGateObservationBoundaryPreflightGateTarget                                  string `json:"admission_final_gate_observation_boundary_preflight_gate_target"`
	AdmissionFinalGateObservationBoundaryPreflightGateTargetKind                              string `json:"admission_final_gate_observation_boundary_preflight_gate_target_kind"`
	AdmissionFinalGateObservationBoundaryPreflightGateTargetMode                              string `json:"admission_final_gate_observation_boundary_preflight_gate_target_mode"`
	AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly                              bool   `json:"admission_final_gate_observation_boundary_preflight_gate_dry_run_only"`
	AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified                       bool   `json:"admission_final_gate_observation_boundary_preflight_gate_preflight_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified                        bool   `json:"admission_final_gate_observation_boundary_preflight_gate_boundary_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified                     bool   `json:"admission_final_gate_observation_boundary_preflight_gate_observation_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified                        bool   `json:"admission_final_gate_observation_boundary_preflight_gate_read_back_verified"`
	AdmissionFinalGateObservationBoundaryPreflightGateReady                                   bool   `json:"admission_final_gate_observation_boundary_preflight_gate_ready"`
	FinalGateObservationBoundaryPreflightGateKind                                             string `json:"final_gate_observation_boundary_preflight_gate_kind"`
	FinalGateObservationBoundaryPreflightGateMode                                             string `json:"final_gate_observation_boundary_preflight_gate_mode"`
	FinalGateObservationBoundaryPreflightGateStage                                            string `json:"final_gate_observation_boundary_preflight_gate_stage"`
	FinalGateObservationBoundaryPreflightGateRawDreamTextObserved                             bool   `json:"final_gate_observation_boundary_preflight_gate_raw_dream_text_observed"`
	FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded                            bool   `json:"final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded"`
	FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed                              bool   `json:"final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed"`
	FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed                              bool   `json:"final_gate_observation_boundary_preflight_gate_janus_surface_allowed"`
	FinalGateObservationBoundaryPreflightGateCoocLearningAllowed                              bool   `json:"final_gate_observation_boundary_preflight_gate_cooc_learning_allowed"`
	FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed                              bool   `json:"final_gate_observation_boundary_preflight_gate_delta_harvest_allowed"`
	FinalGateObservationBoundaryPreflightGateBodyMutationAllowed                              bool   `json:"final_gate_observation_boundary_preflight_gate_body_mutation_allowed"`
	FinalGateObservationBoundaryPreflightGatePreStateHashRequired                             bool   `json:"final_gate_observation_boundary_preflight_gate_pre_state_hash_required"`
	FinalGateObservationBoundaryPreflightGatePostStateHashRequired                            bool   `json:"final_gate_observation_boundary_preflight_gate_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady    bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightConsumed     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightRequired     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate    bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID       string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id"`
	AdmissionFinalGateObservationBoundaryPreflightGateHash                                    string `json:"admission_final_gate_observation_boundary_preflight_gate_hash"`
	AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash                            string `json:"admission_final_gate_observation_boundary_preflight_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausal string `json:"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_causal_id"`
	SourceAdmissionFinalGateObservationBoundaryPreflightHash                                  string `json:"source_admission_final_gate_observation_boundary_preflight_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash                          string `json:"source_admission_final_gate_observation_boundary_preflight_read_back_hash"`
	SourceAdmissionFinalGateObservationBoundaryPreflightReceiptShape                          string `json:"source_admission_final_gate_observation_boundary_preflight_receipt_shape"`
	SourceAdmissionFinalGateObservationBoundaryPreflightState                                 string `json:"source_admission_final_gate_observation_boundary_preflight_state"`
	SourceAdmissionFinalGateObservationBoundaryPreflightAction                                string `json:"source_admission_final_gate_observation_boundary_preflight_action"`
	SourceAdmissionFinalGateObservationBoundaryPreflightTarget                                string `json:"source_admission_final_gate_observation_boundary_preflight_target"`
	SourceAdmissionFinalGateObservationBoundaryPreflightTargetKind                            string `json:"source_admission_final_gate_observation_boundary_preflight_target_kind"`
	SourceAdmissionFinalGateObservationBoundaryPreflightTargetMode                            string `json:"source_admission_final_gate_observation_boundary_preflight_target_mode"`
	SourceAdmissionFinalGateObservationBoundaryPreflightDryRunOnly                            bool   `json:"source_admission_final_gate_observation_boundary_preflight_dry_run_only"`
	SourceAdmissionFinalGateObservationBoundaryPreflightBoundaryVerified                      bool   `json:"source_admission_final_gate_observation_boundary_preflight_boundary_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightObservationVerified                   bool   `json:"source_admission_final_gate_observation_boundary_preflight_observation_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightReadBackVerified                      bool   `json:"source_admission_final_gate_observation_boundary_preflight_read_back_verified"`
	SourceAdmissionFinalGateObservationBoundaryPreflightReady                                 bool   `json:"source_admission_final_gate_observation_boundary_preflight_ready"`
	SourceFinalGateObservationBoundaryPreflightKind                                           string `json:"source_final_gate_observation_boundary_preflight_kind"`
	SourceFinalGateObservationBoundaryPreflightMode                                           string `json:"source_final_gate_observation_boundary_preflight_mode"`
	SourceFinalGateObservationBoundaryPreflightStage                                          string `json:"source_final_gate_observation_boundary_preflight_stage"`
	SourceFinalGateObservationBoundaryPreflightRawDreamTextObserved                           bool   `json:"source_final_gate_observation_boundary_preflight_raw_dream_text_observed"`
	SourceFinalGateObservationBoundaryPreflightRawDreamTextForwarded                          bool   `json:"source_final_gate_observation_boundary_preflight_raw_dream_text_forwarded"`
	SourceFinalGateObservationBoundaryPreflightRawDreamTextAllowed                            bool   `json:"source_final_gate_observation_boundary_preflight_raw_dream_text_allowed"`
	SourceFinalGateObservationBoundaryPreflightJanusSurfaceAllowed                            bool   `json:"source_final_gate_observation_boundary_preflight_janus_surface_allowed"`
	SourceFinalGateObservationBoundaryPreflightCoocLearningAllowed                            bool   `json:"source_final_gate_observation_boundary_preflight_cooc_learning_allowed"`
	SourceFinalGateObservationBoundaryPreflightDeltaHarvestAllowed                            bool   `json:"source_final_gate_observation_boundary_preflight_delta_harvest_allowed"`
	SourceFinalGateObservationBoundaryPreflightBodyMutationAllowed                            bool   `json:"source_final_gate_observation_boundary_preflight_body_mutation_allowed"`
	SourceFinalGateObservationBoundaryPreflightPreStateHashRequired                           bool   `json:"source_final_gate_observation_boundary_preflight_pre_state_hash_required"`
	SourceFinalGateObservationBoundaryPreflightPostStateHashRequired                          bool   `json:"source_final_gate_observation_boundary_preflight_post_state_hash_required"`
	SourceAdmissionFinalGateObservationBoundaryPreflightReason                                string `json:"source_admission_final_gate_observation_boundary_preflight_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT")
	}
	preflightPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate output path missing")
	}
	sourcePreflight, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReportForAssert(preflightPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReportError(sourcePreflight, root); err != nil {
		return err
	}
	gate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReport: sourcePreflight,
		AdmissionFinalGateObservationBoundaryPreflightGateState:                                               "blocked",
		AdmissionFinalGateObservationBoundaryPreflightGateAction:                                              "gate_blocked_final_gate_observation_boundary_preflight",
		AdmissionFinalGateObservationBoundaryPreflightGateTarget:                                              "resonance",
		AdmissionFinalGateObservationBoundaryPreflightGateTargetKind:                                          "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight",
		AdmissionFinalGateObservationBoundaryPreflightGateTargetMode:                                          "closed_preflight_gate_guard_dry_run",
		AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly:                                          true,
		AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified:                                   true,
		AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified:                                    sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified,
		AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified:                                 sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightObservationVerified,
		AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified:                                    true,
		AdmissionFinalGateObservationBoundaryPreflightGateReady:                                               false,
		FinalGateObservationBoundaryPreflightGateKind:                                                         "blocked_final_gate_observation_boundary_preflight_gate",
		FinalGateObservationBoundaryPreflightGateMode:                                                         "no_mutation_preflight_gate",
		FinalGateObservationBoundaryPreflightGateStage:                                                        "post_boundary_preflight_pre_live_admission",
		FinalGateObservationBoundaryPreflightGateRawDreamTextObserved:                                         false,
		FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded:                                        false,
		FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed:                                          false,
		FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed:                                          false,
		FinalGateObservationBoundaryPreflightGateCoocLearningAllowed:                                          false,
		FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed:                                          false,
		FinalGateObservationBoundaryPreflightGateBodyMutationAllowed:                                          false,
		FinalGateObservationBoundaryPreflightGatePreStateHashRequired:                                         true,
		FinalGateObservationBoundaryPreflightGatePostStateHashRequired:                                        true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady:                true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightConsumed:                 true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightRequired:                 true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate:                true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID:                 sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady:              sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausal:             sourcePreflight.CausalID,
		SourceAdmissionFinalGateObservationBoundaryPreflightHash:                                              sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash:                                      sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash,
		SourceAdmissionFinalGateObservationBoundaryPreflightReceiptShape:                                      sourcePreflight.ReceiptShape,
		SourceAdmissionFinalGateObservationBoundaryPreflightState:                                             sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightState,
		SourceAdmissionFinalGateObservationBoundaryPreflightAction:                                            sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightAction,
		SourceAdmissionFinalGateObservationBoundaryPreflightTarget:                                            sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightTarget,
		SourceAdmissionFinalGateObservationBoundaryPreflightTargetKind:                                        sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightTargetKind,
		SourceAdmissionFinalGateObservationBoundaryPreflightTargetMode:                                        sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightTargetMode,
		SourceAdmissionFinalGateObservationBoundaryPreflightDryRunOnly:                                        sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly,
		SourceAdmissionFinalGateObservationBoundaryPreflightBoundaryVerified:                                  sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightObservationVerified:                               sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightObservationVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightReadBackVerified:                                  sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightReadBackVerified,
		SourceAdmissionFinalGateObservationBoundaryPreflightReady:                                             sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightReady,
		SourceFinalGateObservationBoundaryPreflightKind:                                                       sourcePreflight.FinalGateObservationBoundaryPreflightKind,
		SourceFinalGateObservationBoundaryPreflightMode:                                                       sourcePreflight.FinalGateObservationBoundaryPreflightMode,
		SourceFinalGateObservationBoundaryPreflightStage:                                                      sourcePreflight.FinalGateObservationBoundaryPreflightStage,
		SourceFinalGateObservationBoundaryPreflightRawDreamTextObserved:                                       sourcePreflight.FinalGateObservationBoundaryPreflightRawDreamTextObserved,
		SourceFinalGateObservationBoundaryPreflightRawDreamTextForwarded:                                      sourcePreflight.FinalGateObservationBoundaryPreflightRawDreamTextForwarded,
		SourceFinalGateObservationBoundaryPreflightRawDreamTextAllowed:                                        sourcePreflight.FinalGateObservationBoundaryPreflightRawDreamTextAllowed,
		SourceFinalGateObservationBoundaryPreflightJanusSurfaceAllowed:                                        sourcePreflight.FinalGateObservationBoundaryPreflightJanusSurfaceAllowed,
		SourceFinalGateObservationBoundaryPreflightCoocLearningAllowed:                                        sourcePreflight.FinalGateObservationBoundaryPreflightCoocLearningAllowed,
		SourceFinalGateObservationBoundaryPreflightDeltaHarvestAllowed:                                        sourcePreflight.FinalGateObservationBoundaryPreflightDeltaHarvestAllowed,
		SourceFinalGateObservationBoundaryPreflightBodyMutationAllowed:                                        sourcePreflight.FinalGateObservationBoundaryPreflightBodyMutationAllowed,
		SourceFinalGateObservationBoundaryPreflightPreStateHashRequired:                                       sourcePreflight.FinalGateObservationBoundaryPreflightPreStateHashRequired,
		SourceFinalGateObservationBoundaryPreflightPostStateHashRequired:                                      sourcePreflight.FinalGateObservationBoundaryPreflightPostStateHashRequired,
		SourceAdmissionFinalGateObservationBoundaryPreflightReason:                                            sourcePreflight.Reason,
	}
	gate.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema
	gate.Status = "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run"
	gate.Target = "live_route_admission_next_step"
	gate.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate"
	gate.TargetMode = "closed_preflight_gate_guard_dry_run"
	gate.Action = "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run"
	gate.WriterAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate"
	gate.RollbackAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate"
	gate.LedgerState = "blocked"
	gate.LedgerAction = "reject_blocked_admission_final_gate_observation_boundary_preflight_gate"
	gate.LedgerContract = "none"
	gate.LedgerEntrypoint = "none"
	gate.LedgerReceiptShape = "none"
	gate.LedgerWriteScope = "none"
	gate.LedgerReady = false
	gate.LedgerAppendAllowed = false
	gate.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_receipt"
	gate.SourceSchema = sourcePreflight.Schema
	gate.SourceStatus = sourcePreflight.Status
	gate.SourceTarget = sourcePreflight.Target
	gate.SourceReport = preflightPath
	gate.AuthorityGranted = false
	gate.Reason = "weighted resonance shadow graft admission final gate observation boundary preflight gate checked from blocked preflight; live admission remains closed"
	gate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID(gate)
	gate.AdmissionFinalGateObservationBoundaryPreflightGateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateHash(gate)
	gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash(gate)
	gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID(gate)
	if gate.CausalID == "" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateHash == "" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash == "" ||
		gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID == "" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateHash == gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(gate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate] pass: resonance_graft_admission_final_gate_observation_boundary_preflight_gate_report=%s resonance_graft_admission_final_gate_observation_boundary_preflight_report=%s\n", outputPath, preflightPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate")
	}
	if report.TargetMode != "closed_preflight_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate target_mode mismatch: got %q want %q", report.TargetMode, "closed_preflight_gate_guard_dry_run")
	}
	if report.Action != "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate action mismatch: got %q want %q", report.Action, "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" || report.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate ledger guard mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateState != "blocked" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateAction != "gate_blocked_final_gate_observation_boundary_preflight" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateTarget != "resonance" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight" ||
		report.AdmissionFinalGateObservationBoundaryPreflightGateTargetMode != "closed_preflight_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_gate_dry_run_only", report.AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly},
		{"admission_final_gate_observation_boundary_preflight_gate_preflight_verified", report.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_boundary_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_observation_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified},
		{"admission_final_gate_observation_boundary_preflight_gate_read_back_verified", report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified},
		{"final_gate_observation_boundary_preflight_gate_pre_state_hash_required", report.FinalGateObservationBoundaryPreflightGatePreStateHashRequired},
		{"final_gate_observation_boundary_preflight_gate_post_state_hash_required", report.FinalGateObservationBoundaryPreflightGatePostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady},
		{"source_admission_final_gate_observation_boundary_preflight_dry_run_only", report.SourceAdmissionFinalGateObservationBoundaryPreflightDryRunOnly},
		{"source_admission_final_gate_observation_boundary_preflight_boundary_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightBoundaryVerified},
		{"source_admission_final_gate_observation_boundary_preflight_observation_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightObservationVerified},
		{"source_admission_final_gate_observation_boundary_preflight_read_back_verified", report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackVerified},
		{"source_final_gate_observation_boundary_preflight_pre_state_hash_required", report.SourceFinalGateObservationBoundaryPreflightPreStateHashRequired},
		{"source_final_gate_observation_boundary_preflight_post_state_hash_required", report.SourceFinalGateObservationBoundaryPreflightPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady},
		{"admission_final_gate_observation_boundary_preflight_dry_run_only", report.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly},
		{"admission_final_gate_observation_boundary_preflight_boundary_verified", report.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified},
		{"admission_final_gate_observation_boundary_preflight_observation_verified", report.AdmissionFinalGateObservationBoundaryPreflightObservationVerified},
		{"admission_final_gate_observation_boundary_preflight_read_back_verified", report.AdmissionFinalGateObservationBoundaryPreflightReadBackVerified},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady},
		{"weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_boundary_preflight_gate_ready", report.AdmissionFinalGateObservationBoundaryPreflightGateReady},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_observed", report.FinalGateObservationBoundaryPreflightGateRawDreamTextObserved},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_forwarded", report.FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded},
		{"final_gate_observation_boundary_preflight_gate_raw_dream_text_allowed", report.FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed},
		{"final_gate_observation_boundary_preflight_gate_janus_surface_allowed", report.FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed},
		{"final_gate_observation_boundary_preflight_gate_cooc_learning_allowed", report.FinalGateObservationBoundaryPreflightGateCoocLearningAllowed},
		{"final_gate_observation_boundary_preflight_gate_delta_harvest_allowed", report.FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed},
		{"final_gate_observation_boundary_preflight_gate_body_mutation_allowed", report.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed},
		{"source_admission_final_gate_observation_boundary_preflight_ready", report.SourceAdmissionFinalGateObservationBoundaryPreflightReady},
		{"source_final_gate_observation_boundary_preflight_raw_dream_text_observed", report.SourceFinalGateObservationBoundaryPreflightRawDreamTextObserved},
		{"source_final_gate_observation_boundary_preflight_raw_dream_text_forwarded", report.SourceFinalGateObservationBoundaryPreflightRawDreamTextForwarded},
		{"source_final_gate_observation_boundary_preflight_raw_dream_text_allowed", report.SourceFinalGateObservationBoundaryPreflightRawDreamTextAllowed},
		{"source_final_gate_observation_boundary_preflight_janus_surface_allowed", report.SourceFinalGateObservationBoundaryPreflightJanusSurfaceAllowed},
		{"source_final_gate_observation_boundary_preflight_cooc_learning_allowed", report.SourceFinalGateObservationBoundaryPreflightCoocLearningAllowed},
		{"source_final_gate_observation_boundary_preflight_delta_harvest_allowed", report.SourceFinalGateObservationBoundaryPreflightDeltaHarvestAllowed},
		{"source_final_gate_observation_boundary_preflight_body_mutation_allowed", report.SourceFinalGateObservationBoundaryPreflightBodyMutationAllowed},
		{"admission_final_gate_observation_boundary_preflight_ready", report.AdmissionFinalGateObservationBoundaryPreflightReady},
		{"final_gate_observation_boundary_preflight_raw_dream_text_observed", report.FinalGateObservationBoundaryPreflightRawDreamTextObserved},
		{"final_gate_observation_boundary_preflight_raw_dream_text_forwarded", report.FinalGateObservationBoundaryPreflightRawDreamTextForwarded},
		{"final_gate_observation_boundary_preflight_raw_dream_text_allowed", report.FinalGateObservationBoundaryPreflightRawDreamTextAllowed},
		{"final_gate_observation_boundary_preflight_janus_surface_allowed", report.FinalGateObservationBoundaryPreflightJanusSurfaceAllowed},
		{"final_gate_observation_boundary_preflight_cooc_learning_allowed", report.FinalGateObservationBoundaryPreflightCoocLearningAllowed},
		{"final_gate_observation_boundary_preflight_delta_harvest_allowed", report.FinalGateObservationBoundaryPreflightDeltaHarvestAllowed},
		{"final_gate_observation_boundary_preflight_body_mutation_allowed", report.FinalGateObservationBoundaryPreflightBodyMutationAllowed},
		{"admission_final_gate_observation_boundary_ready", report.AdmissionFinalGateObservationBoundaryReady},
		{"admission_final_gate_observation_ready", report.AdmissionFinalGateObservationReady},
		{"admission_final_gate_receiver_ready", report.AdmissionFinalGateReceiverReady},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate opened %s", closed.name)
		}
	}
	if report.FinalGateObservationBoundaryPreflightGateKind != "blocked_final_gate_observation_boundary_preflight_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate gate_kind mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateKind, "blocked_final_gate_observation_boundary_preflight_gate")
	}
	if report.FinalGateObservationBoundaryPreflightGateMode != "no_mutation_preflight_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate gate_mode mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateMode, "no_mutation_preflight_gate")
	}
	if report.FinalGateObservationBoundaryPreflightGateStage != "post_boundary_preflight_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate gate_stage mismatch: got %q want %q", report.FinalGateObservationBoundaryPreflightGateStage, "post_boundary_preflight_pre_live_admission")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_observation_boundary_preflight_gate_hash", report.AdmissionFinalGateObservationBoundaryPreflightGateHash},
		{"admission_final_gate_observation_boundary_preflight_gate_read_back_hash", report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausal},
		{"source_admission_final_gate_observation_boundary_preflight_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightHash},
		{"source_admission_final_gate_observation_boundary_preflight_read_back_hash", report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash},
		{"source_admission_final_gate_observation_boundary_preflight_reason", report.SourceAdmissionFinalGateObservationBoundaryPreflightReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_observation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_receipt" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightState != "blocked" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightAction != "check_blocked_final_gate_observation_boundary_preflight" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightTarget != "resonance" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary" ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightTargetMode != "closed_preflight_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source admission final gate observation boundary preflight shape mismatch")
	}
	if report.SourceFinalGateObservationBoundaryPreflightKind != "blocked_final_gate_observation_boundary_preflight" ||
		report.SourceFinalGateObservationBoundaryPreflightMode != "no_mutation_preflight" ||
		report.SourceFinalGateObservationBoundaryPreflightStage != "post_observation_boundary_pre_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source final gate observation boundary preflight mismatch")
	}
	if report.SourceAdmissionFinalGateObservationBoundaryPreflightReason != "weighted resonance shadow graft admission final gate observation boundary preflight checked from blocked boundary; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source_admission_final_gate_observation_boundary_preflight_reason mismatch: got %q", report.SourceAdmissionFinalGateObservationBoundaryPreflightReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausal, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash, "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-read-") ||
		report.SourceAdmissionFinalGateObservationBoundaryPreflightHash == report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate source preflight proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate causal_id mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate gate_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash == "" || report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate read_back_hash mismatch")
	}
	if report.AdmissionFinalGateObservationBoundaryPreflightGateHash == report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate checked from blocked preflight; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport) string {
	h := hashJSON(struct {
		SourcePreflightID   string `json:"source_admission_final_gate_observation_boundary_preflight_id"`
		SourcePreflightRead string `json:"source_admission_final_gate_observation_boundary_preflight_read_back_hash"`
		SourceReport        string `json:"source_report"`
		GateKind            string `json:"gate_kind"`
		GateStage           string `json:"gate_stage"`
	}{
		SourcePreflightID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourcePreflightRead: report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash,
		SourceReport:        report.SourceReport,
		GateKind:            report.FinalGateObservationBoundaryPreflightGateKind,
		GateStage:           report.FinalGateObservationBoundaryPreflightGateStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourcePreflightID   string `json:"source_admission_final_gate_observation_boundary_preflight_id"`
		SourcePreflightHash string `json:"source_admission_final_gate_observation_boundary_preflight_hash"`
		SourcePreflightRead string `json:"source_admission_final_gate_observation_boundary_preflight_read_back_hash"`
		GateMode            string `json:"gate_mode"`
		PreflightVerified   bool   `json:"preflight_verified"`
		BoundaryVerified    bool   `json:"boundary_verified"`
		ObservationVerified bool   `json:"observation_verified"`
		DryRunOnly          bool   `json:"dry_run_only"`
		RawDreamTextVisible bool   `json:"raw_dream_text_visible"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
	}{
		CausalID:            report.CausalID,
		SourcePreflightID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		SourcePreflightHash: report.SourceAdmissionFinalGateObservationBoundaryPreflightHash,
		SourcePreflightRead: report.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash,
		GateMode:            report.FinalGateObservationBoundaryPreflightGateMode,
		PreflightVerified:   report.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		BoundaryVerified:    report.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		ObservationVerified: report.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		DryRunOnly:          report.AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		RawDreamTextVisible: report.FinalGateObservationBoundaryPreflightGateRawDreamTextObserved || report.FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded,
		BodyMutationAllowed: report.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport) string {
	h := hashJSON(struct {
		GateHash        string `json:"gate_hash"`
		SourcePreflight string `json:"source_admission_final_gate_observation_boundary_preflight_id"`
		GateKind        string `json:"gate_kind"`
		ReadBack        bool   `json:"read_back_verified"`
		GateReady       bool   `json:"gate_ready"`
		AdmissionOpened bool   `json:"admission_opened"`
	}{
		GateHash:        report.AdmissionFinalGateObservationBoundaryPreflightGateHash,
		SourcePreflight: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		GateKind:        report.FinalGateObservationBoundaryPreflightGateKind,
		ReadBack:        report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified,
		GateReady:       report.AdmissionFinalGateObservationBoundaryPreflightGateReady,
		AdmissionOpened: report.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourcePreflightID      string `json:"source_admission_final_gate_observation_boundary_preflight_id"`
		CausalID               string `json:"causal_id"`
		GateHash               string `json:"gate_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"weighted_gate_ready"`
		GateReady              bool   `json:"admission_final_gate_observation_boundary_preflight_gate_ready"`
		PreflightVerified      bool   `json:"preflight_verified"`
		BoundaryVerified       bool   `json:"boundary_verified"`
		ObservationVerified    bool   `json:"observation_verified"`
		DryRunOnly             bool   `json:"dry_run_only"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation_boundary_preflight_gate"`
		SourcePreflightReady   bool   `json:"source_weighted_preflight_ready"`
		SourcePreflightClosed  bool   `json:"source_preflight_closed"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourcePreflightID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID,
		CausalID:               report.CausalID,
		GateHash:               report.AdmissionFinalGateObservationBoundaryPreflightGateHash,
		ReadBackHash:           report.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady,
		GateReady:              report.AdmissionFinalGateObservationBoundaryPreflightGateReady,
		PreflightVerified:      report.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified,
		BoundaryVerified:       report.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified,
		ObservationVerified:    report.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified,
		DryRunOnly:             report.AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly,
		BodyMutationAllowed:    report.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate,
		SourcePreflightReady:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady,
		SourcePreflightClosed:  !report.SourceAdmissionFinalGateObservationBoundaryPreflightReady && !report.SourceFinalGateObservationBoundaryPreflightBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation boundary preflight gate decode failed: %w", err)
	}
	return report, root, nil
}
