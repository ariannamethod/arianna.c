package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_observation.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReport

	AdmissionFinalGateObservationState                                    string `json:"admission_final_gate_observation_state"`
	AdmissionFinalGateObservationAction                                   string `json:"admission_final_gate_observation_action"`
	AdmissionFinalGateObservationTarget                                   string `json:"admission_final_gate_observation_target"`
	AdmissionFinalGateObservationTargetKind                               string `json:"admission_final_gate_observation_target_kind"`
	AdmissionFinalGateObservationTargetMode                               string `json:"admission_final_gate_observation_target_mode"`
	AdmissionFinalGateObservationDryRunOnly                               bool   `json:"admission_final_gate_observation_dry_run_only"`
	AdmissionFinalGateObservationAppendOnly                               bool   `json:"admission_final_gate_observation_append_only"`
	AdmissionFinalGateObservationReadBack                                 bool   `json:"admission_final_gate_observation_read_back"`
	AdmissionFinalGateObservationReceiptVerified                          bool   `json:"admission_final_gate_observation_receipt_verified"`
	AdmissionFinalGateObservationReceiverVerified                         bool   `json:"admission_final_gate_observation_receiver_verified"`
	AdmissionFinalGateObservationReady                                    bool   `json:"admission_final_gate_observation_ready"`
	FinalGateObservationObserver                                          string `json:"final_gate_observation_observer"`
	FinalGateObservationObserverKind                                      string `json:"final_gate_observation_observer_kind"`
	FinalGateObservationKind                                              string `json:"final_gate_observation_kind"`
	FinalGateObservationMode                                              string `json:"final_gate_observation_mode"`
	FinalGateObservationRawDreamTextObserved                              bool   `json:"final_gate_observation_raw_dream_text_observed"`
	FinalGateObservationRawDreamTextForwarded                             bool   `json:"final_gate_observation_raw_dream_text_forwarded"`
	FinalGateObservationRawDreamTextAllowed                               bool   `json:"final_gate_observation_raw_dream_text_allowed"`
	FinalGateObservationJanusSurfaceAllowed                               bool   `json:"final_gate_observation_janus_surface_allowed"`
	FinalGateObservationCoocLearningAllowed                               bool   `json:"final_gate_observation_cooc_learning_allowed"`
	FinalGateObservationDeltaHarvestAllowed                               bool   `json:"final_gate_observation_delta_harvest_allowed"`
	FinalGateObservationBodyMutationAllowed                               bool   `json:"final_gate_observation_body_mutation_allowed"`
	FinalGateObservationPreStateHashRequired                              bool   `json:"final_gate_observation_pre_state_hash_required"`
	FinalGateObservationPostStateHashRequired                             bool   `json:"final_gate_observation_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_observation_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_receiver_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverRequired     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_receiver_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservation     bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID        string `json:"weighted_admission_resonance_graft_admission_final_gate_observation_id"`
	AdmissionFinalGateObservationAppendHash                               string `json:"admission_final_gate_observation_append_hash"`
	AdmissionFinalGateObservationReadBackHash                             string `json:"admission_final_gate_observation_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID     string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady  bool   `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal string `json:"source_weighted_admission_resonance_graft_admission_final_gate_receiver_causal_id"`
	SourceAdmissionFinalGateReceiverPreStateHash                          string `json:"source_admission_final_gate_receiver_pre_state_hash"`
	SourceAdmissionFinalGateReceiverPostStateHash                         string `json:"source_admission_final_gate_receiver_post_state_hash"`
	SourceAdmissionFinalGateReceiverStateDeltaHash                        string `json:"source_admission_final_gate_receiver_state_delta_hash"`
	SourceAdmissionFinalGateReceiverReceiptShape                          string `json:"source_admission_final_gate_receiver_receipt_shape"`
	SourceAdmissionFinalGateReceiverState                                 string `json:"source_admission_final_gate_receiver_state"`
	SourceAdmissionFinalGateReceiverAction                                string `json:"source_admission_final_gate_receiver_action"`
	SourceAdmissionFinalGateReceiverTarget                                string `json:"source_admission_final_gate_receiver_target"`
	SourceAdmissionFinalGateReceiverTargetKind                            string `json:"source_admission_final_gate_receiver_target_kind"`
	SourceAdmissionFinalGateReceiverTargetMode                            string `json:"source_admission_final_gate_receiver_target_mode"`
	SourceAdmissionFinalGateReceiverDryRunOnly                            bool   `json:"source_admission_final_gate_receiver_dry_run_only"`
	SourceAdmissionFinalGateReceiverIntentVerified                        bool   `json:"source_admission_final_gate_receiver_intent_verified"`
	SourceAdmissionFinalGateReceiverFinalGateVerified                     bool   `json:"source_admission_final_gate_receiver_final_gate_verified"`
	SourceAdmissionFinalGateReceiverReady                                 bool   `json:"source_admission_final_gate_receiver_ready"`
	SourceFinalGateReceiver                                               string `json:"source_final_gate_receiver"`
	SourceFinalGateReceiverKind                                           string `json:"source_final_gate_receiver_kind"`
	SourceFinalGateReceiverInfluenceKind                                  string `json:"source_final_gate_receiver_influence_kind"`
	SourceFinalGateReceiverStateHashMode                                  string `json:"source_final_gate_receiver_state_hash_mode"`
	SourceFinalGateReceiverRawDreamTextObserved                           bool   `json:"source_final_gate_receiver_raw_dream_text_observed"`
	SourceFinalGateReceiverRawDreamTextForwarded                          bool   `json:"source_final_gate_receiver_raw_dream_text_forwarded"`
	SourceFinalGateReceiverRawDreamTextAllowed                            bool   `json:"source_final_gate_receiver_raw_dream_text_allowed"`
	SourceFinalGateReceiverJanusSurfaceAllowed                            bool   `json:"source_final_gate_receiver_janus_surface_allowed"`
	SourceFinalGateReceiverCoocLearningAllowed                            bool   `json:"source_final_gate_receiver_cooc_learning_allowed"`
	SourceFinalGateReceiverDeltaHarvestAllowed                            bool   `json:"source_final_gate_receiver_delta_harvest_allowed"`
	SourceFinalGateReceiverBodyMutationAllowed                            bool   `json:"source_final_gate_receiver_body_mutation_allowed"`
	SourceFinalGateReceiverPreStateHashRequired                           bool   `json:"source_final_gate_receiver_pre_state_hash_required"`
	SourceFinalGateReceiverPostStateHashRequired                          bool   `json:"source_final_gate_receiver_post_state_hash_required"`
	SourceAdmissionFinalGateReceiverReason                                string `json:"source_admission_final_gate_receiver_reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservation(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation RESONANCE_GRAFT_ADMISSION_FINAL_GATE_RECEIVER_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT")
	}
	receiverPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation output path missing")
	}
	sourceReceiver, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReportForAssert(receiverPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReportError(sourceReceiver, root); err != nil {
		return err
	}
	observation := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReport: sourceReceiver,
		AdmissionFinalGateObservationState:                                                "recorded",
		AdmissionFinalGateObservationAction:                                               "record_blocked_final_gate_receiver_observation",
		AdmissionFinalGateObservationTarget:                                               "resonance",
		AdmissionFinalGateObservationTargetKind:                                           "weighted_internal_world_shadow_graft_admission_final_gate_receiver",
		AdmissionFinalGateObservationTargetMode:                                           "append_only_read_back_dry_run",
		AdmissionFinalGateObservationDryRunOnly:                                           true,
		AdmissionFinalGateObservationAppendOnly:                                           true,
		AdmissionFinalGateObservationReadBack:                                             true,
		AdmissionFinalGateObservationReceiptVerified:                                      true,
		AdmissionFinalGateObservationReceiverVerified:                                     false,
		AdmissionFinalGateObservationReady:                                                false,
		FinalGateObservationObserver:                                                      "resonance",
		FinalGateObservationObserverKind:                                                  "internal_world",
		FinalGateObservationKind:                                                          "blocked_final_gate_receiver_state_proof",
		FinalGateObservationMode:                                                          "sealed_metadata_observation",
		FinalGateObservationRawDreamTextObserved:                                          false,
		FinalGateObservationRawDreamTextForwarded:                                         false,
		FinalGateObservationRawDreamTextAllowed:                                           false,
		FinalGateObservationJanusSurfaceAllowed:                                           false,
		FinalGateObservationCoocLearningAllowed:                                           false,
		FinalGateObservationDeltaHarvestAllowed:                                           false,
		FinalGateObservationBodyMutationAllowed:                                           false,
		FinalGateObservationPreStateHashRequired:                                          true,
		FinalGateObservationPostStateHashRequired:                                         true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady:                 true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed:                 true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverRequired:                 true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservation:                 true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID:                 sourceReceiver.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady:              sourceReceiver.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal:             sourceReceiver.CausalID,
		SourceAdmissionFinalGateReceiverPreStateHash:                                      sourceReceiver.AdmissionFinalGateReceiverPreStateHash,
		SourceAdmissionFinalGateReceiverPostStateHash:                                     sourceReceiver.AdmissionFinalGateReceiverPostStateHash,
		SourceAdmissionFinalGateReceiverStateDeltaHash:                                    sourceReceiver.AdmissionFinalGateReceiverStateDeltaHash,
		SourceAdmissionFinalGateReceiverReceiptShape:                                      sourceReceiver.ReceiptShape,
		SourceAdmissionFinalGateReceiverState:                                             sourceReceiver.AdmissionFinalGateReceiverState,
		SourceAdmissionFinalGateReceiverAction:                                            sourceReceiver.AdmissionFinalGateReceiverAction,
		SourceAdmissionFinalGateReceiverTarget:                                            sourceReceiver.AdmissionFinalGateReceiverTarget,
		SourceAdmissionFinalGateReceiverTargetKind:                                        sourceReceiver.AdmissionFinalGateReceiverTargetKind,
		SourceAdmissionFinalGateReceiverTargetMode:                                        sourceReceiver.AdmissionFinalGateReceiverTargetMode,
		SourceAdmissionFinalGateReceiverDryRunOnly:                                        sourceReceiver.AdmissionFinalGateReceiverDryRunOnly,
		SourceAdmissionFinalGateReceiverIntentVerified:                                    sourceReceiver.AdmissionFinalGateReceiverIntentVerified,
		SourceAdmissionFinalGateReceiverFinalGateVerified:                                 sourceReceiver.AdmissionFinalGateReceiverFinalGateVerified,
		SourceAdmissionFinalGateReceiverReady:                                             sourceReceiver.AdmissionFinalGateReceiverReady,
		SourceFinalGateReceiver:                                                           sourceReceiver.FinalGateReceiver,
		SourceFinalGateReceiverKind:                                                       sourceReceiver.FinalGateReceiverKind,
		SourceFinalGateReceiverInfluenceKind:                                              sourceReceiver.FinalGateReceiverInfluenceKind,
		SourceFinalGateReceiverStateHashMode:                                              sourceReceiver.FinalGateReceiverStateHashMode,
		SourceFinalGateReceiverRawDreamTextObserved:                                       sourceReceiver.FinalGateReceiverRawDreamTextObserved,
		SourceFinalGateReceiverRawDreamTextForwarded:                                      sourceReceiver.FinalGateReceiverRawDreamTextForwarded,
		SourceFinalGateReceiverRawDreamTextAllowed:                                        sourceReceiver.FinalGateReceiverRawDreamTextAllowed,
		SourceFinalGateReceiverJanusSurfaceAllowed:                                        sourceReceiver.FinalGateReceiverJanusSurfaceAllowed,
		SourceFinalGateReceiverCoocLearningAllowed:                                        sourceReceiver.FinalGateReceiverCoocLearningAllowed,
		SourceFinalGateReceiverDeltaHarvestAllowed:                                        sourceReceiver.FinalGateReceiverDeltaHarvestAllowed,
		SourceFinalGateReceiverBodyMutationAllowed:                                        sourceReceiver.FinalGateReceiverBodyMutationAllowed,
		SourceFinalGateReceiverPreStateHashRequired:                                       sourceReceiver.FinalGateReceiverPreStateHashRequired,
		SourceFinalGateReceiverPostStateHashRequired:                                      sourceReceiver.FinalGateReceiverPostStateHashRequired,
		SourceAdmissionFinalGateReceiverReason:                                            sourceReceiver.Reason,
	}
	observation.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema
	observation.Status = "shadow_graft_admission_final_gate_observation_recorded_dry_run"
	observation.Target = "live_route_admission_next_step"
	observation.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_observation"
	observation.TargetMode = "append_only_read_back_dry_run"
	observation.Action = "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run"
	observation.WriterAction = "reject_blocked_admission_final_gate_observation"
	observation.RollbackAction = "reject_blocked_admission_final_gate_observation"
	observation.LedgerState = "blocked"
	observation.LedgerAction = "reject_blocked_admission_final_gate_observation"
	observation.LedgerContract = "none"
	observation.LedgerEntrypoint = "none"
	observation.LedgerReceiptShape = "none"
	observation.LedgerWriteScope = "none"
	observation.LedgerReady = false
	observation.LedgerAppendAllowed = false
	observation.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt"
	observation.SourceSchema = sourceReceiver.Schema
	observation.SourceStatus = sourceReceiver.Status
	observation.SourceTarget = sourceReceiver.Target
	observation.SourceReport = receiverPath
	observation.AuthorityGranted = false
	observation.Reason = "weighted resonance shadow graft admission final gate observation recorded from blocked receiver; live admission remains closed"
	observation.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausalID(observation)
	observation.AdmissionFinalGateObservationAppendHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAppendHash(observation)
	observation.AdmissionFinalGateObservationReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReadBackHash(observation)
	observation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID(observation)
	if observation.CausalID == "" ||
		observation.AdmissionFinalGateObservationAppendHash == "" ||
		observation.AdmissionFinalGateObservationReadBackHash == "" ||
		observation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID == "" ||
		observation.AdmissionFinalGateObservationAppendHash == observation.AdmissionFinalGateObservationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation read-back proof failed")
	}
	raw, err := json.MarshalIndent(observation, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation] pass: resonance_graft_admission_final_gate_observation_report=%s resonance_graft_admission_final_gate_receiver_report=%s\n", outputPath, receiverPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_observation_recorded_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_observation")
	}
	if report.TargetMode != "append_only_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation target_mode mismatch: got %q want %q", report.TargetMode, "append_only_read_back_dry_run")
	}
	if report.Action != "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation action mismatch: got %q want %q", report.Action, "record_weighted_resonance_shadow_graft_admission_final_gate_observation_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_observation" || report.RollbackAction != "reject_blocked_admission_final_gate_observation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_observation" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation ledger guard mismatch")
	}
	if report.AdmissionFinalGateObservationState != "recorded" ||
		report.AdmissionFinalGateObservationAction != "record_blocked_final_gate_receiver_observation" ||
		report.AdmissionFinalGateObservationTarget != "resonance" ||
		report.AdmissionFinalGateObservationTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_receiver" ||
		report.AdmissionFinalGateObservationTargetMode != "append_only_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_observation_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_dry_run_only", report.AdmissionFinalGateObservationDryRunOnly},
		{"admission_final_gate_observation_append_only", report.AdmissionFinalGateObservationAppendOnly},
		{"admission_final_gate_observation_read_back", report.AdmissionFinalGateObservationReadBack},
		{"admission_final_gate_observation_receipt_verified", report.AdmissionFinalGateObservationReceiptVerified},
		{"final_gate_observation_pre_state_hash_required", report.FinalGateObservationPreStateHashRequired},
		{"final_gate_observation_post_state_hash_required", report.FinalGateObservationPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_observation_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_receiver_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReceiverRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_observation", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservation},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady},
		{"source_admission_final_gate_receiver_dry_run_only", report.SourceAdmissionFinalGateReceiverDryRunOnly},
		{"source_final_gate_receiver_pre_state_hash_required", report.SourceFinalGateReceiverPreStateHashRequired},
		{"source_final_gate_receiver_post_state_hash_required", report.SourceFinalGateReceiverPostStateHashRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_observation_receiver_verified", report.AdmissionFinalGateObservationReceiverVerified},
		{"admission_final_gate_observation_ready", report.AdmissionFinalGateObservationReady},
		{"final_gate_observation_raw_dream_text_observed", report.FinalGateObservationRawDreamTextObserved},
		{"final_gate_observation_raw_dream_text_forwarded", report.FinalGateObservationRawDreamTextForwarded},
		{"final_gate_observation_raw_dream_text_allowed", report.FinalGateObservationRawDreamTextAllowed},
		{"final_gate_observation_janus_surface_allowed", report.FinalGateObservationJanusSurfaceAllowed},
		{"final_gate_observation_cooc_learning_allowed", report.FinalGateObservationCoocLearningAllowed},
		{"final_gate_observation_delta_harvest_allowed", report.FinalGateObservationDeltaHarvestAllowed},
		{"final_gate_observation_body_mutation_allowed", report.FinalGateObservationBodyMutationAllowed},
		{"source_admission_final_gate_receiver_intent_verified", report.SourceAdmissionFinalGateReceiverIntentVerified},
		{"source_admission_final_gate_receiver_final_gate_verified", report.SourceAdmissionFinalGateReceiverFinalGateVerified},
		{"source_admission_final_gate_receiver_ready", report.SourceAdmissionFinalGateReceiverReady},
		{"source_final_gate_receiver_raw_dream_text_observed", report.SourceFinalGateReceiverRawDreamTextObserved},
		{"source_final_gate_receiver_raw_dream_text_forwarded", report.SourceFinalGateReceiverRawDreamTextForwarded},
		{"source_final_gate_receiver_raw_dream_text_allowed", report.SourceFinalGateReceiverRawDreamTextAllowed},
		{"source_final_gate_receiver_janus_surface_allowed", report.SourceFinalGateReceiverJanusSurfaceAllowed},
		{"source_final_gate_receiver_cooc_learning_allowed", report.SourceFinalGateReceiverCoocLearningAllowed},
		{"source_final_gate_receiver_delta_harvest_allowed", report.SourceFinalGateReceiverDeltaHarvestAllowed},
		{"source_final_gate_receiver_body_mutation_allowed", report.SourceFinalGateReceiverBodyMutationAllowed},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate observation opened %s", closed.name)
		}
	}
	if report.FinalGateObservationObserver != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation observer mismatch: got %q want %q", report.FinalGateObservationObserver, "resonance")
	}
	if report.FinalGateObservationObserverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation observer_kind mismatch: got %q want %q", report.FinalGateObservationObserverKind, "internal_world")
	}
	if report.FinalGateObservationKind != "blocked_final_gate_receiver_state_proof" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation observation_kind mismatch: got %q want %q", report.FinalGateObservationKind, "blocked_final_gate_receiver_state_proof")
	}
	if report.FinalGateObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation observation_mode mismatch: got %q want %q", report.FinalGateObservationMode, "sealed_metadata_observation")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_observation_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_observation_append_hash", report.AdmissionFinalGateObservationAppendHash},
		{"admission_final_gate_observation_read_back_hash", report.AdmissionFinalGateObservationReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_receiver_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal},
		{"source_admission_final_gate_receiver_pre_state_hash", report.SourceAdmissionFinalGateReceiverPreStateHash},
		{"source_admission_final_gate_receiver_post_state_hash", report.SourceAdmissionFinalGateReceiverPostStateHash},
		{"source_admission_final_gate_receiver_state_delta_hash", report.SourceAdmissionFinalGateReceiverStateDeltaHash},
		{"source_admission_final_gate_receiver_reason", report.SourceAdmissionFinalGateReceiverReason},
		{"source_weighted_admission_resonance_graft_admission_final_gate_intent_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID},
		{"source_weighted_admission_resonance_graft_admission_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSealID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate observation %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_receiver_previewed_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_receiver_previewed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionFinalGateReceiverReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_receiver_receipt" ||
		report.SourceAdmissionFinalGateReceiverState != "previewed" ||
		report.SourceAdmissionFinalGateReceiverAction != "preview_blocked_final_gate_receiver" ||
		report.SourceAdmissionFinalGateReceiverTarget != "resonance" ||
		report.SourceAdmissionFinalGateReceiverTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_intent" ||
		report.SourceAdmissionFinalGateReceiverTargetMode != "bounded_receiver_preview_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source admission final gate receiver shape mismatch")
	}
	if report.SourceFinalGateReceiver != "resonance" ||
		report.SourceFinalGateReceiverKind != "internal_world" ||
		report.SourceFinalGateReceiverInfluenceKind != "bounded_direction" ||
		report.SourceFinalGateReceiverStateHashMode != "blocked_intent_receiver_preview" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source final gate receiver mismatch")
	}
	if report.SourceAdmissionFinalGateReceiverReason != "weighted resonance shadow graft admission final gate receiver previewed from blocked final gate intent; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source_admission_final_gate_receiver_reason mismatch: got %q", report.SourceAdmissionFinalGateReceiverReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID, "weighted-resonance-graft-admission-final-gate-receiver-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal, "weighted-resonance-graft-admission-final-gate-receiver-causal-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateReceiverPreStateHash, "weighted-resonance-graft-admission-final-gate-receiver-pre-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateReceiverPostStateHash, "weighted-resonance-graft-admission-final-gate-receiver-post-") ||
		!strings.HasPrefix(report.SourceAdmissionFinalGateReceiverStateDeltaHash, "weighted-resonance-graft-admission-final-gate-receiver-delta-") ||
		report.SourceAdmissionFinalGateReceiverPreStateHash == report.SourceAdmissionFinalGateReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation source final gate receiver state proof mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation causal_id mismatch")
	}
	if report.AdmissionFinalGateObservationAppendHash == "" || report.AdmissionFinalGateObservationAppendHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAppendHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation append_hash mismatch")
	}
	if report.AdmissionFinalGateObservationReadBackHash == "" || report.AdmissionFinalGateObservationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation read_back_hash mismatch")
	}
	if report.AdmissionFinalGateObservationAppendHash == report.AdmissionFinalGateObservationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate observation recorded from blocked receiver; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate observation reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport) string {
	h := hashJSON(struct {
		SourceReceiverID     string `json:"source_admission_final_gate_receiver_id"`
		SourceReceiverCausal string `json:"source_admission_final_gate_receiver_causal_id"`
		SourceReport         string `json:"source_report"`
		Observer             string `json:"observer"`
		ObserverKind         string `json:"observer_kind"`
		ObservationKind      string `json:"observation_kind"`
	}{
		SourceReceiverID:     report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		SourceReceiverCausal: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverCausal,
		SourceReport:         report.SourceReport,
		Observer:             report.FinalGateObservationObserver,
		ObserverKind:         report.FinalGateObservationObserverKind,
		ObservationKind:      report.FinalGateObservationKind,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationAppendHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourceReceiverID    string `json:"source_admission_final_gate_receiver_id"`
		ReceiverPreHash     string `json:"receiver_pre_hash"`
		ReceiverPostHash    string `json:"receiver_post_hash"`
		ReceiverDeltaHash   string `json:"receiver_delta_hash"`
		ObservationMode     string `json:"observation_mode"`
		AppendOnly          bool   `json:"append_only"`
		DryRunOnly          bool   `json:"dry_run_only"`
		RawDreamTextVisible bool   `json:"raw_dream_text_visible"`
	}{
		CausalID:            report.CausalID,
		SourceReceiverID:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		ReceiverPreHash:     report.SourceAdmissionFinalGateReceiverPreStateHash,
		ReceiverPostHash:    report.SourceAdmissionFinalGateReceiverPostStateHash,
		ReceiverDeltaHash:   report.SourceAdmissionFinalGateReceiverStateDeltaHash,
		ObservationMode:     report.FinalGateObservationMode,
		AppendOnly:          report.AdmissionFinalGateObservationAppendOnly,
		DryRunOnly:          report.AdmissionFinalGateObservationDryRunOnly,
		RawDreamTextVisible: report.FinalGateObservationRawDreamTextObserved || report.FinalGateObservationRawDreamTextForwarded,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-append-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport) string {
	h := hashJSON(struct {
		AppendHash      string `json:"append_hash"`
		SourceReceiver  string `json:"source_admission_final_gate_receiver_id"`
		ObservationKind string `json:"observation_kind"`
		ReadBack        bool   `json:"read_back"`
		ReceiptVerified bool   `json:"receipt_verified"`
		BodyMutation    bool   `json:"body_mutation"`
	}{
		AppendHash:      report.AdmissionFinalGateObservationAppendHash,
		SourceReceiver:  report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		ObservationKind: report.FinalGateObservationKind,
		ReadBack:        report.AdmissionFinalGateObservationReadBack,
		ReceiptVerified: report.AdmissionFinalGateObservationReceiptVerified,
		BodyMutation:    report.FinalGateObservationBodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReceiverID       string `json:"source_admission_final_gate_receiver_id"`
		CausalID               string `json:"causal_id"`
		AppendHash             string `json:"append_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"weighted_observation_ready"`
		ObservationReady       bool   `json:"admission_final_gate_observation_ready"`
		AppendOnly             bool   `json:"append_only"`
		ReadBack               bool   `json:"read_back"`
		ReceiptVerified        bool   `json:"receipt_verified"`
		DryRunOnly             bool   `json:"dry_run_only"`
		RawDreamTextObserved   bool   `json:"raw_dream_text_observed"`
		RawDreamTextForwarded  bool   `json:"raw_dream_text_forwarded"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_observation"`
		SourceReceiverReady    bool   `json:"source_weighted_receiver_ready"`
		SourceIntentReady      bool   `json:"source_weighted_intent_ready"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceReceiverID:       report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverID,
		CausalID:               report.CausalID,
		AppendHash:             report.AdmissionFinalGateObservationAppendHash,
		ReadBackHash:           report.AdmissionFinalGateObservationReadBackHash,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady,
		ObservationReady:       report.AdmissionFinalGateObservationReady,
		AppendOnly:             report.AdmissionFinalGateObservationAppendOnly,
		ReadBack:               report.AdmissionFinalGateObservationReadBack,
		ReceiptVerified:        report.AdmissionFinalGateObservationReceiptVerified,
		DryRunOnly:             report.AdmissionFinalGateObservationDryRunOnly,
		RawDreamTextObserved:   report.FinalGateObservationRawDreamTextObserved,
		RawDreamTextForwarded:  report.FinalGateObservationRawDreamTextForwarded,
		BodyMutationAllowed:    report.FinalGateObservationBodyMutationAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservation,
		SourceReceiverReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady,
		SourceIntentReady:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-observation-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate observation decode failed: %w", err)
	}
	return report, root, nil
}
