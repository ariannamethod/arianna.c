package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_final_gate.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport

	AdmissionFinalGateState                                    string `json:"admission_final_gate_state"`
	AdmissionFinalGateAction                                   string `json:"admission_final_gate_action"`
	AdmissionFinalGateTarget                                   string `json:"admission_final_gate_target"`
	AdmissionFinalGateTargetKind                               string `json:"admission_final_gate_target_kind"`
	AdmissionFinalGateTargetMode                               string `json:"admission_final_gate_target_mode"`
	AdmissionFinalGateDryRunOnly                               bool   `json:"admission_final_gate_dry_run_only"`
	AdmissionFinalGateSealVerified                             bool   `json:"admission_final_gate_seal_verified"`
	AdmissionFinalGateAuthorityVerified                        bool   `json:"admission_final_gate_authority_verified"`
	AdmissionFinalGatePermitVerified                           bool   `json:"admission_final_gate_permit_verified"`
	AdmissionFinalGateLedgerVerified                           bool   `json:"admission_final_gate_ledger_verified"`
	AdmissionFinalGateReady                                    bool   `json:"admission_final_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateReady     bool   `json:"weighted_admission_resonance_graft_admission_final_gate_ready"`
	WeightedAdmissionResonanceGraftAdmissionSealConsumed       bool   `json:"weighted_admission_resonance_graft_admission_seal_consumed"`
	WeightedAdmissionResonanceGraftAdmissionSealRequired       bool   `json:"weighted_admission_resonance_graft_admission_seal_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionFinalGate     bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate"`
	WeightedAdmissionResonanceGraftAdmissionFinalGateID        string `json:"weighted_admission_resonance_graft_admission_final_gate_id"`
	AdmissionFinalGateHash                                     string `json:"admission_final_gate_hash"`
	AdmissionFinalGateReadBackHash                             string `json:"admission_final_gate_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealID       string `json:"source_weighted_admission_resonance_graft_admission_seal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReady    bool   `json:"source_weighted_admission_resonance_graft_admission_seal_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealCausalID string `json:"source_weighted_admission_resonance_graft_admission_seal_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealHash     string `json:"source_weighted_admission_resonance_graft_admission_seal_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack string `json:"source_weighted_admission_resonance_graft_admission_seal_read_back_hash"`
	SourceAdmissionSealReportReceiptShape                      string `json:"source_admission_seal_report_receipt_shape"`
	SourceAdmissionSealState                                   string `json:"source_admission_seal_state"`
	SourceAdmissionSealAction                                  string `json:"source_admission_seal_action"`
	SourceAdmissionSealTarget                                  string `json:"source_admission_seal_target"`
	SourceAdmissionSealTargetKind                              string `json:"source_admission_seal_target_kind"`
	SourceAdmissionSealTargetMode                              string `json:"source_admission_seal_target_mode"`
	SourceAdmissionSealDryRunOnly                              bool   `json:"source_admission_seal_dry_run_only"`
	SourceAdmissionSealAuthorityVerified                       bool   `json:"source_admission_seal_authority_verified"`
	SourceAdmissionSealPermitVerified                          bool   `json:"source_admission_seal_permit_verified"`
	SourceAdmissionSealLedgerVerified                          bool   `json:"source_admission_seal_ledger_verified"`
	SourceAdmissionSealReady                                   bool   `json:"source_admission_seal_ready"`
	SourceAdmissionSealImmutableReceipt                        bool   `json:"source_admission_seal_immutable_receipt"`
	SourceAdmissionSealReason                                  string `json:"source_admission_seal_reason"`
	SourceAdmissionAuthoritySchema                             string `json:"source_admission_authority_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate RESONANCE_GRAFT_ADMISSION_SEAL_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_REPORT")
	}
	sealPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission final gate output path missing")
	}
	sourceSeal, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportForAssert(sealPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportError(sourceSeal, root); err != nil {
		return err
	}
	finalGate := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport: sourceSeal,
		AdmissionFinalGateState:                                    "blocked",
		AdmissionFinalGateAction:                                   "reject_blocked_admission_seal",
		AdmissionFinalGateTarget:                                   "live_admission_final_gate",
		AdmissionFinalGateTargetKind:                               "weighted_internal_world_shadow_graft_admission_seal",
		AdmissionFinalGateTargetMode:                               "closed_final_gate_guard_dry_run",
		AdmissionFinalGateDryRunOnly:                               true,
		AdmissionFinalGateSealVerified:                             false,
		AdmissionFinalGateAuthorityVerified:                        false,
		AdmissionFinalGatePermitVerified:                           false,
		AdmissionFinalGateLedgerVerified:                           false,
		AdmissionFinalGateReady:                                    false,
		WeightedAdmissionResonanceGraftAdmissionFinalGateReady:     true,
		WeightedAdmissionResonanceGraftAdmissionSealConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionSealRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionFinalGate:     true,
		SourceWeightedAdmissionResonanceGraftAdmissionSealID:       sourceSeal.WeightedAdmissionResonanceGraftAdmissionSealID,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReady:    sourceSeal.WeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSealCausalID: sourceSeal.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSealHash:     sourceSeal.AdmissionSealHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack: sourceSeal.AdmissionSealReadBackHash,
		SourceAdmissionSealReportReceiptShape:                      sourceSeal.ReceiptShape,
		SourceAdmissionSealState:                                   sourceSeal.AdmissionSealState,
		SourceAdmissionSealAction:                                  sourceSeal.AdmissionSealAction,
		SourceAdmissionSealTarget:                                  sourceSeal.AdmissionSealTarget,
		SourceAdmissionSealTargetKind:                              sourceSeal.AdmissionSealTargetKind,
		SourceAdmissionSealTargetMode:                              sourceSeal.AdmissionSealTargetMode,
		SourceAdmissionSealDryRunOnly:                              sourceSeal.AdmissionSealDryRunOnly,
		SourceAdmissionSealAuthorityVerified:                       sourceSeal.AdmissionSealAuthorityVerified,
		SourceAdmissionSealPermitVerified:                          sourceSeal.AdmissionSealPermitVerified,
		SourceAdmissionSealLedgerVerified:                          sourceSeal.AdmissionSealLedgerVerified,
		SourceAdmissionSealReady:                                   sourceSeal.AdmissionSealReady,
		SourceAdmissionSealImmutableReceipt:                        sourceSeal.AdmissionSealImmutableReceipt,
		SourceAdmissionSealReason:                                  sourceSeal.Reason,
		SourceAdmissionAuthoritySchema:                             sourceSeal.SourceSchema,
	}
	finalGate.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema
	finalGate.Status = "shadow_graft_admission_final_gate_blocked_dry_run"
	finalGate.TargetKind = "weighted_internal_world_shadow_graft_admission_final_gate"
	finalGate.TargetMode = "closed_final_gate_guard_dry_run"
	finalGate.Action = "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run"
	finalGate.WriterAction = "reject_blocked_admission_seal"
	finalGate.RollbackAction = "reject_blocked_admission_seal"
	finalGate.LedgerState = "blocked"
	finalGate.LedgerAction = "reject_blocked_admission_seal"
	finalGate.LedgerContract = "none"
	finalGate.LedgerEntrypoint = "none"
	finalGate.LedgerReceiptShape = "none"
	finalGate.LedgerWriteScope = "none"
	finalGate.LedgerReady = false
	finalGate.LedgerAppendAllowed = false
	finalGate.ReceiptShape = "weighted_resonance_shadow_graft_admission_final_gate_receipt"
	finalGate.SourceSchema = sourceSeal.Schema
	finalGate.SourceStatus = sourceSeal.Status
	finalGate.SourceTarget = sourceSeal.Target
	finalGate.SourceReport = sealPath
	finalGate.AuthorityGranted = false
	finalGate.Reason = "weighted resonance shadow graft admission final gate blocked by blocked seal; final admission remains closed"
	finalGate.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID(finalGate)
	finalGate.AdmissionFinalGateHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateHash(finalGate)
	finalGate.AdmissionFinalGateReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReadBackHash(finalGate)
	finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateID(finalGate)
	if finalGate.CausalID == "" ||
		finalGate.AdmissionFinalGateHash == "" ||
		finalGate.AdmissionFinalGateReadBackHash == "" ||
		finalGate.WeightedAdmissionResonanceGraftAdmissionFinalGateID == "" ||
		finalGate.AdmissionFinalGateHash == finalGate.AdmissionFinalGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate read-back proof failed")
	}
	raw, err := json.MarshalIndent(finalGate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission final gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-final-gate] pass: resonance_graft_admission_final_gate_report=%s resonance_graft_admission_seal_report=%s\n", outputPath, sealPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission final gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateSchema)
	}
	if report.Status != "shadow_graft_admission_final_gate_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate status mismatch: got %q want %q", report.Status, "shadow_graft_admission_final_gate_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate" {
		return fmt.Errorf("weighted admission resonance graft admission final gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_final_gate")
	}
	if report.TargetMode != "closed_final_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate target_mode mismatch: got %q want %q", report.TargetMode, "closed_final_gate_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_seal_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_seal" || report.RollbackAction != "reject_blocked_admission_seal" {
		return fmt.Errorf("weighted admission resonance graft admission final gate writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_seal" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate ledger guard mismatch")
	}
	if report.AdmissionFinalGateState != "blocked" ||
		report.AdmissionFinalGateAction != "reject_blocked_admission_seal" ||
		report.AdmissionFinalGateTarget != "live_admission_final_gate" ||
		report.AdmissionFinalGateTargetKind != "weighted_internal_world_shadow_graft_admission_seal" ||
		report.AdmissionFinalGateTargetMode != "closed_final_gate_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission final gate receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_final_gate_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_ready", report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady},
		{"weighted_admission_resonance_graft_admission_seal_consumed", report.WeightedAdmissionResonanceGraftAdmissionSealConsumed},
		{"weighted_admission_resonance_graft_admission_seal_required", report.WeightedAdmissionResonanceGraftAdmissionSealRequired},
		{"next_step_blocked_without_resonance_graft_admission_final_gate", report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate},
		{"source_weighted_admission_resonance_graft_admission_seal_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady},
		{"weighted_admission_resonance_graft_admission_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionSealReady},
		{"weighted_admission_resonance_graft_admission_authority_consumed", report.WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed},
		{"weighted_admission_resonance_graft_admission_authority_required", report.WeightedAdmissionResonanceGraftAdmissionAuthorityRequired},
		{"weighted_admission_resonance_graft_admission_authority_ready", report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady},
		{"weighted_admission_resonance_graft_admission_permit_consumed", report.WeightedAdmissionResonanceGraftAdmissionPermitConsumed},
		{"weighted_admission_resonance_graft_admission_permit_required", report.WeightedAdmissionResonanceGraftAdmissionPermitRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_final_gate_seal_verified", report.AdmissionFinalGateSealVerified},
		{"admission_final_gate_authority_verified", report.AdmissionFinalGateAuthorityVerified},
		{"admission_final_gate_permit_verified", report.AdmissionFinalGatePermitVerified},
		{"admission_final_gate_ledger_verified", report.AdmissionFinalGateLedgerVerified},
		{"admission_final_gate_ready", report.AdmissionFinalGateReady},
		{"source_admission_seal_authority_verified", report.SourceAdmissionSealAuthorityVerified},
		{"source_admission_seal_permit_verified", report.SourceAdmissionSealPermitVerified},
		{"source_admission_seal_ledger_verified", report.SourceAdmissionSealLedgerVerified},
		{"source_admission_seal_ready", report.SourceAdmissionSealReady},
		{"admission_seal_authority_verified", report.AdmissionSealAuthorityVerified},
		{"admission_seal_permit_verified", report.AdmissionSealPermitVerified},
		{"admission_seal_ledger_verified", report.AdmissionSealLedgerVerified},
		{"admission_seal_ready", report.AdmissionSealReady},
		{"source_admission_authority_permit_verified", report.SourceAdmissionAuthorityPermitVerified},
		{"source_admission_authority_ledger_verified", report.SourceAdmissionAuthorityLedgerVerified},
		{"source_admission_authority_writer_ready", report.SourceAdmissionAuthorityWriterReady},
		{"source_admission_authority_rollback_ready", report.SourceAdmissionAuthorityRollbackReady},
		{"source_admission_authority_ready", report.SourceAdmissionAuthorityReady},
		{"source_admission_authority_granted", report.SourceAdmissionAuthorityGranted},
		{"admission_authority_permit_verified", report.AdmissionAuthorityPermitVerified},
		{"admission_authority_ledger_verified", report.AdmissionAuthorityLedgerVerified},
		{"admission_authority_writer_ready", report.AdmissionAuthorityWriterReady},
		{"admission_authority_rollback_ready", report.AdmissionAuthorityRollbackReady},
		{"admission_authority_ready", report.AdmissionAuthorityReady},
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
			return fmt.Errorf("weighted admission resonance graft admission final gate opened %s", closed.name)
		}
	}
	if !report.AdmissionFinalGateDryRunOnly || !report.AdmissionSealDryRunOnly || !report.SourceAdmissionSealDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission final gate dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_final_gate_id", report.WeightedAdmissionResonanceGraftAdmissionFinalGateID},
		{"causal_id", report.CausalID},
		{"admission_final_gate_hash", report.AdmissionFinalGateHash},
		{"admission_final_gate_read_back_hash", report.AdmissionFinalGateReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_seal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSealID},
		{"source_weighted_admission_resonance_graft_admission_seal_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSealCausalID},
		{"source_weighted_admission_resonance_graft_admission_seal_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSealHash},
		{"source_weighted_admission_resonance_graft_admission_seal_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack},
		{"source_admission_seal_reason", report.SourceAdmissionSealReason},
		{"source_admission_authority_schema", report.SourceAdmissionAuthoritySchema},
		{"source_weighted_admission_resonance_graft_admission_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID},
		{"source_weighted_admission_resonance_graft_admission_permit_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission final gate %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission final gate source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_seal_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission final gate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionAuthoritySchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema {
		return fmt.Errorf("weighted admission resonance graft admission final gate source_admission_authority_schema mismatch: got %q want %q", report.SourceAdmissionAuthoritySchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema)
	}
	if report.SourceAdmissionSealReportReceiptShape != "weighted_resonance_shadow_graft_admission_seal_receipt" ||
		report.SourceAdmissionSealState != "sealed" ||
		report.SourceAdmissionSealAction != "seal_blocked_admission_authority" ||
		report.SourceAdmissionSealTarget != "live_admission_authority" ||
		report.SourceAdmissionSealTargetKind != "weighted_internal_world_shadow_graft_admission_authority" ||
		report.SourceAdmissionSealTargetMode != "closed_seal_guard_dry_run" ||
		!report.SourceAdmissionSealImmutableReceipt {
		return fmt.Errorf("weighted admission resonance graft admission final gate source admission seal shape mismatch")
	}
	if report.SourceAdmissionSealReason != "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate source_admission_seal_reason mismatch: got %q", report.SourceAdmissionSealReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSealID, "weighted-resonance-graft-admission-seal-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSealCausalID, "weighted-resonance-graft-admission-seal-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSealHash, "weighted-resonance-graft-admission-seal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack, "weighted-resonance-graft-admission-seal-read-") {
		return fmt.Errorf("weighted admission resonance graft admission final gate source seal mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission final gate body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate causal_id mismatch")
	}
	if report.AdmissionFinalGateHash == "" || report.AdmissionFinalGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate admission_final_gate_hash mismatch")
	}
	if report.AdmissionFinalGateReadBackHash == "" || report.AdmissionFinalGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate admission_final_gate_read_back_hash mismatch")
	}
	if report.AdmissionFinalGateHash == report.AdmissionFinalGateReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission final gate read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionFinalGateID == "" || report.WeightedAdmissionResonanceGraftAdmissionFinalGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateID(report) {
		return fmt.Errorf("weighted admission resonance graft admission final gate id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission final gate blocked by blocked seal; final admission remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission final gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport) string {
	h := hashJSON(struct {
		SourceSealID      string `json:"source_admission_seal_id"`
		SourceSealRead    string `json:"source_admission_seal_read_back_hash"`
		SourceAuthorityID string `json:"source_admission_authority_id"`
		Target            string `json:"target"`
		State             string `json:"admission_final_gate_state"`
		Action            string `json:"admission_final_gate_action"`
	}{
		SourceSealID:      report.SourceWeightedAdmissionResonanceGraftAdmissionSealID,
		SourceSealRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack,
		SourceAuthorityID: report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID,
		Target:            report.Target,
		State:             report.AdmissionFinalGateState,
		Action:            report.AdmissionFinalGateAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealHash         string `json:"source_admission_seal_hash"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		State                  string `json:"admission_final_gate_state"`
		Action                 string `json:"admission_final_gate_action"`
		Target                 string `json:"admission_final_gate_target"`
		TargetKind             string `json:"admission_final_gate_target_kind"`
		TargetMode             string `json:"admission_final_gate_target_mode"`
		DryRunOnly             bool   `json:"admission_final_gate_dry_run_only"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		AuthorityVerified      bool   `json:"admission_final_gate_authority_verified"`
		PermitVerified         bool   `json:"admission_final_gate_permit_verified"`
		LedgerVerified         bool   `json:"admission_final_gate_ledger_verified"`
		Ready                  bool   `json:"admission_final_gate_ready"`
		WeightedReady          bool   `json:"weighted_final_gate_ready"`
		SourceWeightedReady    bool   `json:"source_weighted_seal_ready"`
		SourceAdmissionSeal    bool   `json:"source_admission_seal_ready"`
		SourceSealAuthority    bool   `json:"source_admission_seal_authority_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate"`
	}{
		CausalID:               report.CausalID,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionSealID,
		SourceSealHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionSealHash,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack,
		State:                  report.AdmissionFinalGateState,
		Action:                 report.AdmissionFinalGateAction,
		Target:                 report.AdmissionFinalGateTarget,
		TargetKind:             report.AdmissionFinalGateTargetKind,
		TargetMode:             report.AdmissionFinalGateTargetMode,
		DryRunOnly:             report.AdmissionFinalGateDryRunOnly,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		AuthorityVerified:      report.AdmissionFinalGateAuthorityVerified,
		PermitVerified:         report.AdmissionFinalGatePermitVerified,
		LedgerVerified:         report.AdmissionFinalGateLedgerVerified,
		Ready:                  report.AdmissionFinalGateReady,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceAdmissionSeal:    report.SourceAdmissionSealReady,
		SourceSealAuthority:    report.SourceAdmissionSealAuthorityVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport) string {
	h := hashJSON(struct {
		AdmissionFinalGateHash string `json:"admission_final_gate_hash"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		WeightedReady          bool   `json:"weighted_final_gate_ready"`
		SealConsumed           bool   `json:"seal_consumed"`
		SealRequired           bool   `json:"seal_required"`
		FinalGateReady         bool   `json:"admission_final_gate_ready"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveEnabled            bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		AdmissionFinalGateHash: report.AdmissionFinalGateHash,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionSealID,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		SealConsumed:           report.WeightedAdmissionResonanceGraftAdmissionSealConsumed,
		SealRequired:           report.WeightedAdmissionResonanceGraftAdmissionSealRequired,
		FinalGateReady:         report.AdmissionFinalGateReady,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveEnabled:            report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceSealID           string `json:"source_admission_seal_id"`
		SourceSealHash         string `json:"source_admission_seal_hash"`
		SourceSealRead         string `json:"source_admission_seal_read_back_hash"`
		CausalID               string `json:"causal_id"`
		AdmissionFinalGateHash string `json:"admission_final_gate_hash"`
		AdmissionFinalGateRead string `json:"admission_final_gate_read_back_hash"`
		State                  string `json:"admission_final_gate_state"`
		ActionGate             string `json:"admission_final_gate_action"`
		Ready                  bool   `json:"weighted_final_gate_ready"`
		FinalGateReady         bool   `json:"admission_final_gate_ready"`
		SealVerified           bool   `json:"admission_final_gate_seal_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_final_gate"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceSealID:           report.SourceWeightedAdmissionResonanceGraftAdmissionSealID,
		SourceSealHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionSealHash,
		SourceSealRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionSealReadBack,
		CausalID:               report.CausalID,
		AdmissionFinalGateHash: report.AdmissionFinalGateHash,
		AdmissionFinalGateRead: report.AdmissionFinalGateReadBackHash,
		State:                  report.AdmissionFinalGateState,
		ActionGate:             report.AdmissionFinalGateAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionFinalGateReady,
		FinalGateReady:         report.AdmissionFinalGateReady,
		SealVerified:           report.AdmissionFinalGateSealVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionFinalGate,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-final-gate-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission final gate decode failed: %w", err)
	}
	return report, root, nil
}
