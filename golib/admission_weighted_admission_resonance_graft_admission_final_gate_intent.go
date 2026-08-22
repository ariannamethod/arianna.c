package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate_intent.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport

	AdmissionFinalGateIntentState                                   string  `json:"admission_final_gate_intent_state"`
	AdmissionFinalGateIntentAction                                  string  `json:"admission_final_gate_intent_action"`
	AdmissionFinalGateIntentTarget                                  string  `json:"admission_final_gate_intent_target"`
	AdmissionFinalGateIntentTargetKind                              string  `json:"admission_final_gate_intent_target_kind"`
	AdmissionFinalGateIntentTargetMode                              string  `json:"admission_final_gate_intent_target_mode"`
	AdmissionFinalGateIntentDryRunOnly                              bool    `json:"admission_final_gate_intent_dry_run_only"`
	AdmissionFinalGateIntentFinalGateVerified                       bool    `json:"admission_final_gate_intent_final_gate_verified"`
	AdmissionFinalGateIntentSealVerified                            bool    `json:"admission_final_gate_intent_seal_verified"`
	AdmissionFinalGateIntentReady                                   bool    `json:"admission_final_gate_intent_ready"`
	FinalGateIntentReceiver                                         string  `json:"final_gate_intent_receiver"`
	FinalGateIntentReceiverKind                                     string  `json:"final_gate_intent_receiver_kind"`
	FinalGateIntentInfluenceKind                                    string  `json:"final_gate_intent_influence_kind"`
	FinalGateIntentMaxInfluence                                     float64 `json:"final_gate_intent_max_influence"`
	FinalGateIntentTTLTurns                                         int     `json:"final_gate_intent_ttl_turns"`
	FinalGateIntentRawDreamTextAllowed                              bool    `json:"final_gate_intent_raw_dream_text_allowed"`
	FinalGateIntentJanusSurfaceAllowed                              bool    `json:"final_gate_intent_janus_surface_allowed"`
	FinalGateIntentCoocLearningAllowed                              bool    `json:"final_gate_intent_cooc_learning_allowed"`
	FinalGateIntentDeltaHarvestAllowed                              bool    `json:"final_gate_intent_delta_harvest_allowed"`
	FinalGateIntentPreStateHashRequired                             bool    `json:"final_gate_intent_pre_state_hash_required"`
	FinalGateIntentPostStateHashRequired                            bool    `json:"final_gate_intent_post_state_hash_required"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady    bool    `json:"weighted_admission_resonance_graft_admission_final_gate_intent_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed       bool    `json:"weighted_admission_resonance_graft_admission_final_gate_consumed"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateRequired       bool    `json:"weighted_admission_resonance_graft_admission_final_gate_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent    bool    `json:"next_step_blocked_without_resonance_graft_admission_final_gate_intent"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID       string  `json:"weighted_admission_resonance_graft_admission_final_gate_intent_id"`
	AdmissionFinalGateIntentHash                                    string  `json:"admission_final_gate_intent_hash"`
	AdmissionFinalGateIntentReadBackHash                            string  `json:"admission_final_gate_intent_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID       string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady    bool    `json:"source_weighted_admission_resonance_graft_admission_final_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash     string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack string  `json:"source_weighted_admission_resonance_graft_admission_final_gate_read_back_hash"`
	SourceAdmissionFinalGateReportReceiptShape                      string  `json:"source_admission_final_gate_report_receipt_shape"`
	SourceAdmissionFinalGateState                                   string  `json:"source_admission_final_gate_state"`
	SourceAdmissionFinalGateAction                                  string  `json:"source_admission_final_gate_action"`
	SourceAdmissionFinalGateTarget                                  string  `json:"source_admission_final_gate_target"`
	SourceAdmissionFinalGateTargetKind                              string  `json:"source_admission_final_gate_target_kind"`
	SourceAdmissionFinalGateTargetMode                              string  `json:"source_admission_final_gate_target_mode"`
	SourceAdmissionFinalGateDryRunOnly                              bool    `json:"source_admission_final_gate_dry_run_only"`
	SourceAdmissionFinalGateSealVerified                            bool    `json:"source_admission_final_gate_seal_verified"`
	SourceAdmissionFinalGateAuthorityVerified                       bool    `json:"source_admission_final_gate_authority_verified"`
	SourceAdmissionFinalGatePermitVerified                          bool    `json:"source_admission_final_gate_permit_verified"`
	SourceAdmissionFinalGateLedgerVerified                          bool    `json:"source_admission_final_gate_ledger_verified"`
	SourceAdmissionFinalGateReady                                   bool    `json:"source_admission_final_gate_ready"`
	SourceAdmissionFinalGateReason                                  string  `json:"source_admission_final_gate_reason"`
	SourceAdmissionSealSchema                                       string  `json:"source_admission_seal_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntent(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent RESONANCE_GRAFT_ADMISSION_FINAL_GATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_INTENT_REPORT")
	}
	finalGatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent output path missing")
	}
	sourceFinalGate, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportForAssert(finalGatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportError(sourceFinalGate, root); err != nil {
		return err
	}
	intent := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport: sourceFinalGate,
		AdmissionFinalGateIntentState:                                   "blocked",
		AdmissionFinalGateIntentAction:                                  "draft_blocked_final_gate_intent",
		AdmissionFinalGateIntentTarget:                                  "resonance",
		AdmissionFinalGateIntentTargetKind:                              "weighted_internal_world_shadow_graft_admission_final_gate",
		AdmissionFinalGateIntentTargetMode:                              "bounded_intent_guard_dry_run",
		AdmissionFinalGateIntentDryRunOnly:                              true,
		AdmissionFinalGateIntentFinalGateVerified:                       false,
		AdmissionFinalGateIntentSealVerified:                            false,
		AdmissionFinalGateIntentReady:                                   false,
		FinalGateIntentReceiver:                                         "resonance",
		FinalGateIntentReceiverKind:                                     "internal_world",
		FinalGateIntentInfluenceKind:                                    "bounded_direction",
		FinalGateIntentMaxInfluence:                                     admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain,
		FinalGateIntentTTLTurns:                                         admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL,
		FinalGateIntentRawDreamTextAllowed:                              false,
		FinalGateIntentJanusSurfaceAllowed:                              false,
		FinalGateIntentCoocLearningAllowed:                              false,
		FinalGateIntentDeltaHarvestAllowed:                              false,
		FinalGateIntentPreStateHashRequired:                             true,
		FinalGateIntentPostStateHashRequired:                            true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady:    true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionFinalGateRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent:    true,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID:       sourceFinalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady:    sourceFinalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID: sourceFinalGate.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash:     sourceFinalGate.AdmissionFinalGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack: sourceFinalGate.AdmissionFinalGateReadBackHash,
		SourceAdmissionFinalGateReportReceiptShape:                      sourceFinalGate.ReceiptShape,
		SourceAdmissionFinalGateState:                                   sourceFinalGate.AdmissionFinalGateState,
		SourceAdmissionFinalGateAction:                                  sourceFinalGate.AdmissionFinalGateAction,
		SourceAdmissionFinalGateTarget:                                  sourceFinalGate.AdmissionFinalGateTarget,
		SourceAdmissionFinalGateTargetKind:                              sourceFinalGate.AdmissionFinalGateTargetKind,
		SourceAdmissionFinalGateTargetMode:                              sourceFinalGate.AdmissionFinalGateTargetMode,
		SourceAdmissionFinalGateDryRunOnly:                              sourceFinalGate.AdmissionFinalGateDryRunOnly,
		SourceAdmissionFinalGateSealVerified:                            sourceFinalGate.AdmissionFinalGateSealVerified,
		SourceAdmissionFinalGateAuthorityVerified:                       sourceFinalGate.AdmissionFinalGateAuthorityVerified,
		SourceAdmissionFinalGatePermitVerified:                          sourceFinalGate.AdmissionFinalGatePermitVerified,
		SourceAdmissionFinalGateLedgerVerified:                          sourceFinalGate.AdmissionFinalGateLedgerVerified,
		SourceAdmissionFinalGateReady:                                   sourceFinalGate.AdmissionFinalGateReady,
		SourceAdmissionFinalGateReason:                                  sourceFinalGate.Reason,
		SourceAdmissionSealSchema:                                       sourceFinalGate.SourceSchema,
	}
	intent.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema
	intent.Status = "shadow_graft_admission_final_gate_intent_blocked_dry_run"
	intent.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate_intent"
	intent.TargetMode = "bounded_intent_guard_dry_run"
	intent.Action = "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run"
	intent.WriterAction = "reject_blocked_admission_final_gate_intent"
	intent.RollbackAction = "reject_blocked_admission_final_gate_intent"
	intent.LedgerState = "blocked"
	intent.LedgerAction = "reject_blocked_admission_final_gate_intent"
	intent.LedgerContract = "none"
	intent.LedgerEntrypoint = "none"
	intent.LedgerReceiptShape = "none"
	intent.LedgerWriteScope = "none"
	intent.LedgerReady = false
	intent.LedgerAppendAllowed = false
	intent.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_intent_receipt"
	intent.SourceSchema = sourceFinalGate.Schema
	intent.SourceStatus = sourceFinalGate.Status
	intent.SourceTarget = sourceFinalGate.Target
	intent.SourceReport = finalGatePath
	intent.AuthorityGranted = false
	intent.Reason = "weighted resonance shadow graft admission final gate intent drafted from blocked final gate; live admission remains closed"
	intent.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentCausalID(intent)
	intent.AdmissionFinalGateIntentHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentHash(intent)
	intent.AdmissionFinalGateIntentReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReadBackHash(intent)
	intent.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID(intent)
	if intent.CausalID == "" ||
		intent.AdmissionFinalGateIntentHash == "" ||
		intent.AdmissionFinalGateIntentReadBackHash == "" ||
		intent.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID == "" ||
		intent.AdmissionFinalGateIntentHash == intent.AdmissionFinalGateIntentReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent read-back proof failed")
	}
	raw, err := json.MarshalIndent(intent, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent] pass: resonance_graft_admission_final_gate_intent_report=%s resonance_graft_admission_final_gate_report=%s\n", outputPath, finalGatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-intent-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_intent_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_intent_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_intent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate_intent")
	}
	if report.TargetMode != "bounded_intent_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent target_mode mismatch: got %q want %q", report.TargetMode, "bounded_intent_guard_dry_run")
	}
	if report.Action != "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent action mismatch: got %q want %q", report.Action, "draft_weighted_resonance_shadow_graft_admission_final_gate_intent_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_final_gate_intent" || report.RollbackAction != "reject_blocked_admission_final_gate_intent" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_final_gate_intent" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent ledger guard mismatch")
	}
	if report.AdmissionFinalGateIntentState != "blocked" ||
		report.AdmissionFinalGateIntentAction != "draft_blocked_final_gate_intent" ||
		report.AdmissionFinalGateIntentTarget != "resonance" ||
		report.AdmissionFinalGateIntentTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate" ||
		report.AdmissionFinalGateIntentTargetMode != "bounded_intent_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_intent_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_intent_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_intent_dry_run_only", report.AdmissionFinalGateIntentDryRunOnly},
		{"final_gate_intent_pre_state_hash_required", report.FinalGateIntentPreStateHashRequired},
		{"final_gate_intent_post_state_hash_required", report.FinalGateIntentPostStateHashRequired},
		{"weighted_admission_resonance_graft_admission_final_gate_intent_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady},
		{"weighted_admission_resonance_graft_admission_final_gate_consumed", report.WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed},
		{"weighted_admission_resonance_graft_admission_final_gate_required", report.WeightedAdmissionResonanceGraftAdmissionFinalGateRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate_intent", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent},
		{"source_weighted_admission_resonance_graft_admission_final_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady},
		{"weighted_admission_resonance_graft_admission_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady},
		{"weighted_admission_resonance_graft_admission_seal_consumed", report.WeightedAdmissionResonanceGraftAdmissionSealConsumed},
		{"weighted_admission_resonance_graft_admission_seal_required", report.WeightedAdmissionResonanceGraftAdmissionSealRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate intent %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_intent_final_gate_verified", report.AdmissionFinalGateIntentFinalGateVerified},
		{"admission_final_gate_intent_seal_verified", report.AdmissionFinalGateIntentSealVerified},
		{"admission_final_gate_intent_ready", report.AdmissionFinalGateIntentReady},
		{"final_gate_intent_raw_dream_text_allowed", report.FinalGateIntentRawDreamTextAllowed},
		{"final_gate_intent_janus_surface_allowed", report.FinalGateIntentJanusSurfaceAllowed},
		{"final_gate_intent_cooc_learning_allowed", report.FinalGateIntentCoocLearningAllowed},
		{"final_gate_intent_delta_harvest_allowed", report.FinalGateIntentDeltaHarvestAllowed},
		{"source_admission_final_gate_seal_verified", report.SourceAdmissionFinalGateSealVerified},
		{"source_admission_final_gate_authority_verified", report.SourceAdmissionFinalGateAuthorityVerified},
		{"source_admission_final_gate_permit_verified", report.SourceAdmissionFinalGatePermitVerified},
		{"source_admission_final_gate_ledger_verified", report.SourceAdmissionFinalGateLedgerVerified},
		{"source_admission_final_gate_ready", report.SourceAdmissionFinalGateReady},
		{"admission_final_gate_seal_verified", report.AdmissionFinalGateSealVerified},
		{"admission_final_gate_authority_verified", report.AdmissionFinalGateAuthorityVerified},
		{"admission_final_gate_permit_verified", report.AdmissionFinalGatePermitVerified},
		{"admission_final_gate_ledger_verified", report.AdmissionFinalGateLedgerVerified},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"admission_seal_ready", report.AdmissionSealReady},
		{"admission_authority_granted", report.AdmissionAuthorityGranted},
		{"manual_permit_requested", report.ManualPermitRequested},
		{"permit_key_matched", report.PermitKeyMatched},
		{"admission_permit_ready", report.AdmissionPermitReady},
		{"admission_readiness_ready", report.AdmissionReadinessReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate intent opened %s", closed.name)
		}
	}
	if !report.AdmissionFinalGateIntentDryRunOnly || !report.AdmissionFinalGateDryRunOnly || !report.SourceAdmissionFinalGateDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent dry-run flag mismatch")
	}
	if report.FinalGateIntentReceiver != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent receiver mismatch: got %q want %q", report.FinalGateIntentReceiver, "resonance")
	}
	if report.FinalGateIntentReceiverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent receiver_kind mismatch: got %q want %q", report.FinalGateIntentReceiverKind, "internal_world")
	}
	if report.FinalGateIntentInfluenceKind != "bounded_direction" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent influence_kind mismatch: got %q want %q", report.FinalGateIntentInfluenceKind, "bounded_direction")
	}
	if report.FinalGateIntentMaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent max_influence mismatch: got %.6f want %.6f", report.FinalGateIntentMaxInfluence, admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain)
	}
	if report.FinalGateIntentTTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent ttl_turns mismatch: got %d want %d", report.FinalGateIntentTTLTurns, admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL)
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_intent_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_intent_hash", report.AdmissionFinalGateIntentHash},
		{"admission_final_gate_intent_read_back_hash", report.AdmissionFinalGateIntentReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_final_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID},
		{"source_weighted_admission_resonance_graft_admission_final_gate_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash},
		{"source_weighted_admission_resonance_graft_admission_final_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack},
		{"source_admission_final_gate_reason", report.SourceAdmissionFinalGateReason},
		{"source_admission_seal_schema", report.SourceAdmissionSealSchema},
		{"source_weighted_admission_resonance_graft_admission_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSealID},
		{"source_weighted_admission_resonance_graft_admission_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate intent %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_final_gate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_final_gate_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionSealSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source_admission_seal_schema mismatch: got %q want %q", report.SourceAdmissionSealSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema)
	}
	if report.SourceAdmissionFinalGateReportReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_receipt" ||
		report.SourceAdmissionFinalGateState != "blocked" ||
		report.SourceAdmissionFinalGateAction != "reject_blocked_admission_seal" ||
		report.SourceAdmissionFinalGateTarget != "live_admission_final_gate" ||
		report.SourceAdmissionFinalGateTargetKind != "weighted_internal_world_shadow_graft_admission_seal" ||
		report.SourceAdmissionFinalGateTargetMode != "closed_final_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source admission final gate shape mismatch")
	}
	if report.SourceAdmissionFinalGateReason != "weighted resonance shadow graft admission final gate blocked by blocked seal; final admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source_admission_final_gate_reason mismatch: got %q", report.SourceAdmissionFinalGateReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID, "weighted-resonance-graft-admission-final-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID, "weighted-resonance-graft-admission-final-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash, "weighted-resonance-graft-admission-final-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack, "weighted-resonance-graft-admission-final-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent source final gate mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent causal_id mismatch")
	}
	if report.AdmissionFinalGateIntentHash == "" || report.AdmissionFinalGateIntentHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent admission_final_gate_intent_hash mismatch")
	}
	if report.AdmissionFinalGateIntentReadBackHash == "" || report.AdmissionFinalGateIntentReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent admission_final_gate_intent_read_back_hash mismatch")
	}
	if report.AdmissionFinalGateIntentHash == report.AdmissionFinalGateIntentReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate intent drafted from blocked final gate; live admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate intent reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport) string {
	h := hashJSON(struct {
		SourceFinalGateID   string  `json:"source_admission_final_gate_id"`
		SourceFinalGateRead string  `json:"source_admission_final_gate_read_back_hash"`
		SourceSealID        string  `json:"source_admission_seal_id"`
		Receiver            string  `json:"receiver"`
		ReceiverKind        string  `json:"receiver_kind"`
		InfluenceKind       string  `json:"influence_kind"`
		MaxInfluence        float64 `json:"max_influence"`
		TTLTurns            int     `json:"ttl_turns"`
		State               string  `json:"admission_final_gate_intent_state"`
		Action              string  `json:"admission_final_gate_intent_action"`
	}{
		SourceFinalGateID:   report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID,
		SourceFinalGateRead: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack,
		SourceSealID:        report.SourceWeightedAdmissionResonanceGraftAdmissionSealID,
		Receiver:            report.FinalGateIntentReceiver,
		ReceiverKind:        report.FinalGateIntentReceiverKind,
		InfluenceKind:       report.FinalGateIntentInfluenceKind,
		MaxInfluence:        report.FinalGateIntentMaxInfluence,
		TTLTurns:            report.FinalGateIntentTTLTurns,
		State:               report.AdmissionFinalGateIntentState,
		Action:              report.AdmissionFinalGateIntentAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-intent-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport) string {
	h := hashJSON(struct {
		CausalID               string  `json:"causal_id"`
		SourceFinalGateID      string  `json:"source_admission_final_gate_id"`
		SourceFinalGateHash    string  `json:"source_admission_final_gate_hash"`
		SourceFinalGateRead    string  `json:"source_admission_final_gate_read_back_hash"`
		State                  string  `json:"admission_final_gate_intent_state"`
		Action                 string  `json:"admission_final_gate_intent_action"`
		Target                 string  `json:"admission_final_gate_intent_target"`
		TargetKind             string  `json:"admission_final_gate_intent_target_kind"`
		TargetMode             string  `json:"admission_final_gate_intent_target_mode"`
		DryRunOnly             bool    `json:"admission_final_gate_intent_dry_run_only"`
		Receiver               string  `json:"receiver"`
		ReceiverKind           string  `json:"receiver_kind"`
		InfluenceKind          string  `json:"influence_kind"`
		MaxInfluence           float64 `json:"max_influence"`
		TTLTurns               int     `json:"ttl_turns"`
		FinalGateVerified      bool    `json:"admission_final_gate_intent_final_gate_verified"`
		SealVerified           bool    `json:"admission_final_gate_intent_seal_verified"`
		Ready                  bool    `json:"admission_final_gate_intent_ready"`
		WeightedReady          bool    `json:"weighted_intent_ready"`
		SourceWeightedReady    bool    `json:"source_weighted_final_gate_ready"`
		SourceFinalGateReady   bool    `json:"source_admission_final_gate_ready"`
		RawDreamTextAllowed    bool    `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed    bool    `json:"janus_surface_allowed"`
		CoocLearningAllowed    bool    `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed    bool    `json:"delta_harvest_allowed"`
		WriteAllowed           bool    `json:"write_allowed"`
		AdmissionAllowed       bool    `json:"admission_allowed"`
		LiveAdmissionEnabled   bool    `json:"live_admission_enabled"`
		MutatesState           bool    `json:"mutates_state"`
		BodyMutationAllowed    bool    `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool    `json:"next_step_blocked_without_resonance_graft_admission_final_gate_intent"`
	}{
		CausalID:               report.CausalID,
		SourceFinalGateID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID,
		SourceFinalGateHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash,
		SourceFinalGateRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack,
		State:                  report.AdmissionFinalGateIntentState,
		Action:                 report.AdmissionFinalGateIntentAction,
		Target:                 report.AdmissionFinalGateIntentTarget,
		TargetKind:             report.AdmissionFinalGateIntentTargetKind,
		TargetMode:             report.AdmissionFinalGateIntentTargetMode,
		DryRunOnly:             report.AdmissionFinalGateIntentDryRunOnly,
		Receiver:               report.FinalGateIntentReceiver,
		ReceiverKind:           report.FinalGateIntentReceiverKind,
		InfluenceKind:          report.FinalGateIntentInfluenceKind,
		MaxInfluence:           report.FinalGateIntentMaxInfluence,
		TTLTurns:               report.FinalGateIntentTTLTurns,
		FinalGateVerified:      report.AdmissionFinalGateIntentFinalGateVerified,
		SealVerified:           report.AdmissionFinalGateIntentSealVerified,
		Ready:                  report.AdmissionFinalGateIntentReady,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceFinalGateReady:   report.SourceAdmissionFinalGateReady,
		RawDreamTextAllowed:    report.FinalGateIntentRawDreamTextAllowed,
		JanusSurfaceAllowed:    report.FinalGateIntentJanusSurfaceAllowed,
		CoocLearningAllowed:    report.FinalGateIntentCoocLearningAllowed,
		DeltaHarvestAllowed:    report.FinalGateIntentDeltaHarvestAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-intent-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport) string {
	h := hashJSON(struct {
		IntentHash      string `json:"admission_final_gate_intent_hash"`
		SourceFinalGate string `json:"source_admission_final_gate_id"`
		SourceRead      string `json:"source_admission_final_gate_read_back_hash"`
		WeightedReady   bool   `json:"weighted_intent_ready"`
		FinalConsumed   bool   `json:"final_gate_consumed"`
		FinalRequired   bool   `json:"final_gate_required"`
		IntentReady     bool   `json:"admission_final_gate_intent_ready"`
		FinalVerified   bool   `json:"admission_final_gate_intent_final_gate_verified"`
		WriteAllowed    bool   `json:"write_allowed"`
		Admission       bool   `json:"admission_allowed"`
		LiveEnabled     bool   `json:"live_admission_enabled"`
		MutatesState    bool   `json:"mutates_state"`
	}{
		IntentHash:      report.AdmissionFinalGateIntentHash,
		SourceFinalGate: report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID,
		SourceRead:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack,
		WeightedReady:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		FinalConsumed:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateConsumed,
		FinalRequired:   report.WeightedAdmissionResonanceGraftAdmissionFinalGateRequired,
		IntentReady:     report.AdmissionFinalGateIntentReady,
		FinalVerified:   report.AdmissionFinalGateIntentFinalGateVerified,
		WriteAllowed:    report.WriteAllowed,
		Admission:       report.AdmissionAllowed,
		LiveEnabled:     report.LiveAdmissionEnabled,
		MutatesState:    report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-intent-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceFinalGateID      string `json:"source_admission_final_gate_id"`
		SourceFinalGateHash    string `json:"source_admission_final_gate_hash"`
		SourceFinalGateRead    string `json:"source_admission_final_gate_read_back_hash"`
		CausalID               string `json:"causal_id"`
		IntentHash             string `json:"admission_final_gate_intent_hash"`
		IntentRead             string `json:"admission_final_gate_intent_read_back_hash"`
		State                  string `json:"admission_final_gate_intent_state"`
		ActionIntent           string `json:"admission_final_gate_intent_action"`
		Ready                  bool   `json:"weighted_intent_ready"`
		IntentReady            bool   `json:"admission_final_gate_intent_ready"`
		FinalGateVerified      bool   `json:"admission_final_gate_intent_final_gate_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate_intent"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceFinalGateID:      report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateID,
		SourceFinalGateHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateHash,
		SourceFinalGateRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReadBack,
		CausalID:               report.CausalID,
		IntentHash:             report.AdmissionFinalGateIntentHash,
		IntentRead:             report.AdmissionFinalGateIntentReadBackHash,
		State:                  report.AdmissionFinalGateIntentState,
		ActionIntent:           report.AdmissionFinalGateIntentAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateIntentReady,
		IntentReady:            report.AdmissionFinalGateIntentReady,
		FinalGateVerified:      report.AdmissionFinalGateIntentFinalGateVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateIntent,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-intent-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateIntentReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate intent path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate intent not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate intent not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate intent JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate intent decode failed: %w", err)
	}
	return report, root, nil
}
