package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_permit.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport

	AdmissionPermitState                                      string `json:"admission_permit_state"`
	AdmissionPermitAction                                     string `json:"admission_permit_action"`
	AdmissionPermitTarget                                     string `json:"admission_permit_target"`
	AdmissionPermitTargetKind                                 string `json:"admission_permit_target_kind"`
	AdmissionPermitTargetMode                                 string `json:"admission_permit_target_mode"`
	AdmissionPermitDryRunOnly                                 bool   `json:"admission_permit_dry_run_only"`
	AdmissionPermitReadinessVerified                          bool   `json:"admission_permit_readiness_verified"`
	AdmissionPermitLedgerVerified                             bool   `json:"admission_permit_ledger_verified"`
	AdmissionPermitWriterReady                                bool   `json:"admission_permit_writer_ready"`
	AdmissionPermitRollbackReady                              bool   `json:"admission_permit_rollback_ready"`
	AdmissionPermitLedgerReady                                bool   `json:"admission_permit_ledger_ready"`
	AdmissionPermitReady                                      bool   `json:"admission_permit_ready"`
	ManualPermitRequested                                     bool   `json:"manual_permit_requested"`
	PermitKeyMatched                                          bool   `json:"permit_key_matched"`
	WeightedAdmissionResonanceGraftAdmissionPermitReady       bool   `json:"weighted_admission_resonance_graft_admission_permit_ready"`
	WeightedAdmissionResonanceGraftAdmissionReadinessConsumed bool   `json:"weighted_admission_resonance_graft_admission_readiness_consumed"`
	WeightedAdmissionResonanceGraftAdmissionReadinessRequired bool   `json:"weighted_admission_resonance_graft_admission_readiness_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionPermit       bool   `json:"next_step_blocked_without_resonance_graft_admission_permit"`
	WeightedAdmissionResonanceGraftAdmissionPermitID          string `json:"weighted_admission_resonance_graft_admission_permit_id"`
	AdmissionPermitHash                                       string `json:"admission_permit_hash"`
	AdmissionPermitReadBackHash                               string `json:"admission_permit_read_back_hash"`

	SourceWeightedAdmissionResonanceGraftAdmissionReadinessID       string `json:"source_weighted_admission_resonance_graft_admission_readiness_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady    bool   `json:"source_weighted_admission_resonance_graft_admission_readiness_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessCausalID string `json:"source_weighted_admission_resonance_graft_admission_readiness_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash     string `json:"source_weighted_admission_resonance_graft_admission_readiness_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack string `json:"source_weighted_admission_resonance_graft_admission_readiness_read_back_hash"`
	SourceAdmissionReadinessReportReceiptShape                      string `json:"source_admission_readiness_report_receipt_shape"`
	SourceAdmissionReadinessState                                   string `json:"source_admission_readiness_state"`
	SourceAdmissionReadinessAction                                  string `json:"source_admission_readiness_action"`
	SourceAdmissionReadinessTarget                                  string `json:"source_admission_readiness_target"`
	SourceAdmissionReadinessTargetKind                              string `json:"source_admission_readiness_target_kind"`
	SourceAdmissionReadinessTargetMode                              string `json:"source_admission_readiness_target_mode"`
	SourceAdmissionReadinessDryRunOnly                              bool   `json:"source_admission_readiness_dry_run_only"`
	SourceAdmissionReadinessLedgerVerified                          bool   `json:"source_admission_readiness_ledger_verified"`
	SourceAdmissionReadinessWriterReady                             bool   `json:"source_admission_readiness_writer_ready"`
	SourceAdmissionReadinessRollbackReady                           bool   `json:"source_admission_readiness_rollback_ready"`
	SourceAdmissionReadinessLedgerReady                             bool   `json:"source_admission_readiness_ledger_ready"`
	SourceAdmissionReadinessReady                                   bool   `json:"source_admission_readiness_ready"`
	SourceAdmissionReadinessReason                                  string `json:"source_admission_readiness_reason"`
	SourceLedgerVerificationSchema                                  string `json:"source_ledger_verification_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermit(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-permit RESONANCE_GRAFT_ADMISSION_READINESS_REPORT RESONANCE_GRAFT_ADMISSION_PERMIT_REPORT")
	}
	readinessPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission permit output path missing")
	}
	sourceReadiness, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportForAssert(readinessPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportError(sourceReadiness, root); err != nil {
		return err
	}
	permit := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport: sourceReadiness,
		AdmissionPermitState:                                            "blocked",
		AdmissionPermitAction:                                           "reject_blocked_admission_readiness",
		AdmissionPermitTarget:                                           "live_admission",
		AdmissionPermitTargetKind:                                       "weighted_internal_world_shadow_graft_admission_readiness",
		AdmissionPermitTargetMode:                                       "closed_permit_guard_dry_run",
		AdmissionPermitDryRunOnly:                                       true,
		AdmissionPermitReadinessVerified:                                false,
		AdmissionPermitLedgerVerified:                                   false,
		AdmissionPermitWriterReady:                                      false,
		AdmissionPermitRollbackReady:                                    false,
		AdmissionPermitLedgerReady:                                      false,
		AdmissionPermitReady:                                            false,
		ManualPermitRequested:                                           false,
		PermitKeyMatched:                                                false,
		WeightedAdmissionResonanceGraftAdmissionPermitReady:             true,
		WeightedAdmissionResonanceGraftAdmissionReadinessConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionReadinessRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionPermit:             true,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessID:       sourceReadiness.WeightedAdmissionResonanceGraftAdmissionReadinessID,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady:    sourceReadiness.WeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessCausalID: sourceReadiness.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash:     sourceReadiness.AdmissionReadinessHash,
		SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack: sourceReadiness.AdmissionReadinessReadBackHash,
		SourceAdmissionReadinessReportReceiptShape:                      sourceReadiness.ReceiptShape,
		SourceAdmissionReadinessState:                                   sourceReadiness.AdmissionReadinessState,
		SourceAdmissionReadinessAction:                                  sourceReadiness.AdmissionReadinessAction,
		SourceAdmissionReadinessTarget:                                  sourceReadiness.AdmissionReadinessTarget,
		SourceAdmissionReadinessTargetKind:                              sourceReadiness.AdmissionReadinessTargetKind,
		SourceAdmissionReadinessTargetMode:                              sourceReadiness.AdmissionReadinessTargetMode,
		SourceAdmissionReadinessDryRunOnly:                              sourceReadiness.AdmissionReadinessDryRunOnly,
		SourceAdmissionReadinessLedgerVerified:                          sourceReadiness.AdmissionReadinessLedgerVerified,
		SourceAdmissionReadinessWriterReady:                             sourceReadiness.AdmissionReadinessWriterReady,
		SourceAdmissionReadinessRollbackReady:                           sourceReadiness.AdmissionReadinessRollbackReady,
		SourceAdmissionReadinessLedgerReady:                             sourceReadiness.AdmissionReadinessLedgerReady,
		SourceAdmissionReadinessReady:                                   sourceReadiness.AdmissionReadinessReady,
		SourceAdmissionReadinessReason:                                  sourceReadiness.Reason,
		SourceLedgerVerificationSchema:                                  sourceReadiness.SourceSchema,
	}
	permit.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema
	permit.Status = "shadow_graft_admission_permit_blocked_dry_run"
	permit.TargetKind = "weighted_internal_world_shadow_graft_admission_permit"
	permit.TargetMode = "closed_permit_guard_dry_run"
	permit.Action = "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run"
	permit.WriterAction = "reject_blocked_admission_readiness"
	permit.RollbackAction = "reject_blocked_admission_readiness"
	permit.LedgerState = "blocked"
	permit.LedgerAction = "reject_blocked_admission_readiness"
	permit.LedgerContract = "none"
	permit.LedgerEntrypoint = "none"
	permit.LedgerReceiptShape = "none"
	permit.LedgerWriteScope = "none"
	permit.LedgerReady = false
	permit.LedgerAppendAllowed = false
	permit.ReceiptShape = "weighted_resonance_shadow_graft_admission_permit_receipt"
	permit.SourceSchema = sourceReadiness.Schema
	permit.SourceStatus = sourceReadiness.Status
	permit.SourceTarget = sourceReadiness.Target
	permit.SourceReport = readinessPath
	permit.Reason = "weighted resonance shadow graft admission permit blocked by blocked readiness; manual permit remains closed"
	permit.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitCausalID(permit)
	permit.AdmissionPermitHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitHash(permit)
	permit.AdmissionPermitReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReadBackHash(permit)
	permit.WeightedAdmissionResonanceGraftAdmissionPermitID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitID(permit)
	if permit.CausalID == "" ||
		permit.AdmissionPermitHash == "" ||
		permit.AdmissionPermitReadBackHash == "" ||
		permit.WeightedAdmissionResonanceGraftAdmissionPermitID == "" ||
		permit.AdmissionPermitHash == permit.AdmissionPermitReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission permit read-back proof failed")
	}
	raw, err := json.MarshalIndent(permit, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission permit marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission permit write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-permit] pass: resonance_graft_admission_permit_report=%s resonance_graft_admission_readiness_report=%s\n", outputPath, readinessPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-permit-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission permit schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema {
		return fmt.Errorf("weighted admission resonance graft admission permit schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema)
	}
	if report.Status != "shadow_graft_admission_permit_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit status mismatch: got %q want %q", report.Status, "shadow_graft_admission_permit_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_permit" {
		return fmt.Errorf("weighted admission resonance graft admission permit target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_permit")
	}
	if report.TargetMode != "closed_permit_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit target_mode mismatch: got %q want %q", report.TargetMode, "closed_permit_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_readiness_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_readiness" || report.RollbackAction != "reject_blocked_admission_readiness" {
		return fmt.Errorf("weighted admission resonance graft admission permit writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_readiness" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission permit ledger guard mismatch")
	}
	if report.AdmissionPermitState != "blocked" ||
		report.AdmissionPermitAction != "reject_blocked_admission_readiness" ||
		report.AdmissionPermitTarget != "live_admission" ||
		report.AdmissionPermitTargetKind != "weighted_internal_world_shadow_graft_admission_readiness" ||
		report.AdmissionPermitTargetMode != "closed_permit_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_permit_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission permit receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_permit_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_permit_ready", report.WeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"weighted_admission_resonance_graft_admission_readiness_consumed", report.WeightedAdmissionResonanceGraftAdmissionReadinessConsumed},
		{"weighted_admission_resonance_graft_admission_readiness_required", report.WeightedAdmissionResonanceGraftAdmissionReadinessRequired},
		{"next_step_blocked_without_resonance_graft_admission_permit", report.NextStepBlockedWithoutResonanceGraftAdmissionPermit},
		{"source_weighted_admission_resonance_graft_admission_readiness_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReady},
		{"weighted_admission_resonance_graft_admission_readiness_ready", report.WeightedAdmissionResonanceGraftAdmissionReadinessReady},
		{"weighted_admission_resonance_graft_admission_ledger_verification_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_verification_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired},
		{"weighted_admission_resonance_graft_admission_ledger_verification_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady},
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
			return fmt.Errorf("weighted admission resonance graft admission permit %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"manual_permit_requested", report.ManualPermitRequested},
		{"permit_key_matched", report.PermitKeyMatched},
		{"admission_permit_readiness_verified", report.AdmissionPermitReadinessVerified},
		{"admission_permit_ledger_verified", report.AdmissionPermitLedgerVerified},
		{"admission_permit_writer_ready", report.AdmissionPermitWriterReady},
		{"admission_permit_rollback_ready", report.AdmissionPermitRollbackReady},
		{"admission_permit_ledger_ready", report.AdmissionPermitLedgerReady},
		{"admission_permit_ready", report.AdmissionPermitReady},
		{"source_admission_readiness_ledger_verified", report.SourceAdmissionReadinessLedgerVerified},
		{"source_admission_readiness_writer_ready", report.SourceAdmissionReadinessWriterReady},
		{"source_admission_readiness_rollback_ready", report.SourceAdmissionReadinessRollbackReady},
		{"source_admission_readiness_ledger_ready", report.SourceAdmissionReadinessLedgerReady},
		{"source_admission_readiness_ready", report.SourceAdmissionReadinessReady},
		{"admission_readiness_ledger_verified", report.AdmissionReadinessLedgerVerified},
		{"admission_readiness_writer_ready", report.AdmissionReadinessWriterReady},
		{"admission_readiness_rollback_ready", report.AdmissionReadinessRollbackReady},
		{"admission_readiness_ledger_ready", report.AdmissionReadinessLedgerReady},
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
			return fmt.Errorf("weighted admission resonance graft admission permit opened %s", closed.name)
		}
	}
	if !report.AdmissionPermitDryRunOnly || !report.AdmissionReadinessDryRunOnly || !report.SourceAdmissionReadinessDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission permit dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_permit_id", report.WeightedAdmissionResonanceGraftAdmissionPermitID},
		{"causal_id", report.CausalID},
		{"admission_permit_hash", report.AdmissionPermitHash},
		{"admission_permit_read_back_hash", report.AdmissionPermitReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_readiness_id", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID},
		{"source_weighted_admission_resonance_graft_admission_readiness_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessCausalID},
		{"source_weighted_admission_resonance_graft_admission_readiness_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash},
		{"source_weighted_admission_resonance_graft_admission_readiness_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack},
		{"source_admission_readiness_reason", report.SourceAdmissionReadinessReason},
		{"source_ledger_verification_schema", report.SourceLedgerVerificationSchema},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission permit %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema {
		return fmt.Errorf("weighted admission resonance graft admission permit source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_readiness_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_readiness_blocked_dry_run")
	}
	if report.SourceLedgerVerificationSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema {
		return fmt.Errorf("weighted admission resonance graft admission permit source_ledger_verification_schema mismatch: got %q want %q", report.SourceLedgerVerificationSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema)
	}
	if report.SourceAdmissionReadinessReportReceiptShape != "weighted_resonance_shadow_graft_admission_readiness_receipt" ||
		report.SourceAdmissionReadinessState != "blocked" ||
		report.SourceAdmissionReadinessAction != "reject_blocked_ledger_verification" ||
		report.SourceAdmissionReadinessTarget != "live_admission" ||
		report.SourceAdmissionReadinessTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_verification" ||
		report.SourceAdmissionReadinessTargetMode != "closed_readiness_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission permit source admission readiness shape mismatch")
	}
	if report.SourceAdmissionReadinessReason != "weighted resonance shadow graft admission readiness blocked by blocked ledger verification; live admission readiness remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission permit source_admission_readiness_reason mismatch: got %q", report.SourceAdmissionReadinessReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID, "weighted-resonance-graft-admission-readiness-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessCausalID, "weighted-resonance-graft-admission-readiness-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash, "weighted-resonance-graft-admission-readiness-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack, "weighted-resonance-graft-admission-readiness-read-") {
		return fmt.Errorf("weighted admission resonance graft admission permit source readiness mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission permit body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission permit causal_id mismatch")
	}
	if report.AdmissionPermitHash == "" || report.AdmissionPermitHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission permit admission_permit_hash mismatch")
	}
	if report.AdmissionPermitReadBackHash == "" || report.AdmissionPermitReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission permit admission_permit_read_back_hash mismatch")
	}
	if report.AdmissionPermitHash == report.AdmissionPermitReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission permit read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionPermitID == "" || report.WeightedAdmissionResonanceGraftAdmissionPermitID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitID(report) {
		return fmt.Errorf("weighted admission resonance graft admission permit id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission permit blocked by blocked readiness; manual permit remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission permit reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport) string {
	h := hashJSON(struct {
		SourceReadinessID   string `json:"source_admission_readiness_id"`
		SourceReadinessRead string `json:"source_admission_readiness_read_back_hash"`
		SourceVerification  string `json:"source_ledger_verification_id"`
		Target              string `json:"target"`
		State               string `json:"admission_permit_state"`
		Action              string `json:"admission_permit_action"`
	}{
		SourceReadinessID:   report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID,
		SourceReadinessRead: report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack,
		SourceVerification:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		Target:              report.Target,
		State:               report.AdmissionPermitState,
		Action:              report.AdmissionPermitAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-permit-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceReadinessID      string `json:"source_admission_readiness_id"`
		SourceReadinessHash    string `json:"source_admission_readiness_hash"`
		SourceReadinessRead    string `json:"source_admission_readiness_read_back_hash"`
		State                  string `json:"admission_permit_state"`
		Action                 string `json:"admission_permit_action"`
		Target                 string `json:"admission_permit_target"`
		TargetKind             string `json:"admission_permit_target_kind"`
		TargetMode             string `json:"admission_permit_target_mode"`
		DryRunOnly             bool   `json:"admission_permit_dry_run_only"`
		ReadinessVerified      bool   `json:"admission_permit_readiness_verified"`
		LedgerVerified         bool   `json:"admission_permit_ledger_verified"`
		WriterReady            bool   `json:"admission_permit_writer_ready"`
		RollbackReady          bool   `json:"admission_permit_rollback_ready"`
		LedgerReady            bool   `json:"admission_permit_ledger_ready"`
		Ready                  bool   `json:"admission_permit_ready"`
		ManualRequested        bool   `json:"manual_permit_requested"`
		KeyMatched             bool   `json:"permit_key_matched"`
		WeightedReady          bool   `json:"weighted_permit_ready"`
		SourceReadinessReady   bool   `json:"source_admission_readiness_ready"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_permit"`
	}{
		CausalID:               report.CausalID,
		SourceReadinessID:      report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID,
		SourceReadinessHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash,
		SourceReadinessRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack,
		State:                  report.AdmissionPermitState,
		Action:                 report.AdmissionPermitAction,
		Target:                 report.AdmissionPermitTarget,
		TargetKind:             report.AdmissionPermitTargetKind,
		TargetMode:             report.AdmissionPermitTargetMode,
		DryRunOnly:             report.AdmissionPermitDryRunOnly,
		ReadinessVerified:      report.AdmissionPermitReadinessVerified,
		LedgerVerified:         report.AdmissionPermitLedgerVerified,
		WriterReady:            report.AdmissionPermitWriterReady,
		RollbackReady:          report.AdmissionPermitRollbackReady,
		LedgerReady:            report.AdmissionPermitLedgerReady,
		Ready:                  report.AdmissionPermitReady,
		ManualRequested:        report.ManualPermitRequested,
		KeyMatched:             report.PermitKeyMatched,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceReadinessReady:   report.SourceAdmissionReadinessReady,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionPermit,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-permit-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport) string {
	h := hashJSON(struct {
		AdmissionPermitHash string `json:"admission_permit_hash"`
		SourceReadinessID   string `json:"source_admission_readiness_id"`
		SourceReadinessRead string `json:"source_admission_readiness_read_back_hash"`
		WeightedReady       bool   `json:"weighted_permit_ready"`
		ReadinessConsumed   bool   `json:"readiness_consumed"`
		ReadinessRequired   bool   `json:"readiness_required"`
		PermitReady         bool   `json:"admission_permit_ready"`
		ReadinessVerified   bool   `json:"admission_permit_readiness_verified"`
		ManualRequested     bool   `json:"manual_permit_requested"`
		KeyMatched          bool   `json:"permit_key_matched"`
		WriteAllowed        bool   `json:"write_allowed"`
		AdmissionAllowed    bool   `json:"admission_allowed"`
		LiveEnabled         bool   `json:"live_admission_enabled"`
		MutatesState        bool   `json:"mutates_state"`
	}{
		AdmissionPermitHash: report.AdmissionPermitHash,
		SourceReadinessID:   report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID,
		SourceReadinessRead: report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack,
		WeightedReady:       report.WeightedAdmissionResonanceGraftAdmissionPermitReady,
		ReadinessConsumed:   report.WeightedAdmissionResonanceGraftAdmissionReadinessConsumed,
		ReadinessRequired:   report.WeightedAdmissionResonanceGraftAdmissionReadinessRequired,
		PermitReady:         report.AdmissionPermitReady,
		ReadinessVerified:   report.AdmissionPermitReadinessVerified,
		ManualRequested:     report.ManualPermitRequested,
		KeyMatched:          report.PermitKeyMatched,
		WriteAllowed:        report.WriteAllowed,
		AdmissionAllowed:    report.AdmissionAllowed,
		LiveEnabled:         report.LiveAdmissionEnabled,
		MutatesState:        report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-permit-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReadinessID       string `json:"source_admission_readiness_id"`
		SourceReadinessHash     string `json:"source_admission_readiness_hash"`
		SourceReadinessRead     string `json:"source_admission_readiness_read_back_hash"`
		CausalID                string `json:"causal_id"`
		AdmissionPermitHash     string `json:"admission_permit_hash"`
		AdmissionPermitReadBack string `json:"admission_permit_read_back_hash"`
		State                   string `json:"admission_permit_state"`
		ActionPermit            string `json:"admission_permit_action"`
		Ready                   bool   `json:"weighted_permit_ready"`
		PermitReady             bool   `json:"admission_permit_ready"`
		ManualRequested         bool   `json:"manual_permit_requested"`
		KeyMatched              bool   `json:"permit_key_matched"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_permit"`
	}{
		Schema:                  report.Schema,
		Status:                  report.Status,
		Action:                  report.Action,
		SourceReadinessID:       report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID,
		SourceReadinessHash:     report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessHash,
		SourceReadinessRead:     report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessReadBack,
		CausalID:                report.CausalID,
		AdmissionPermitHash:     report.AdmissionPermitHash,
		AdmissionPermitReadBack: report.AdmissionPermitReadBackHash,
		State:                   report.AdmissionPermitState,
		ActionPermit:            report.AdmissionPermitAction,
		Ready:                   report.WeightedAdmissionResonanceGraftAdmissionPermitReady,
		PermitReady:             report.AdmissionPermitReady,
		ManualRequested:         report.ManualPermitRequested,
		KeyMatched:              report.PermitKeyMatched,
		WriteAllowed:            report.WriteAllowed,
		AdmissionAllowed:        report.AdmissionAllowed,
		LiveAdmissionEnabled:    report.LiveAdmissionEnabled,
		MutatesState:            report.MutatesState,
		NextStepBlockedWithout:  report.NextStepBlockedWithoutResonanceGraftAdmissionPermit,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-permit-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission permit path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission permit not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission permit not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission permit JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission permit decode failed: %w", err)
	}
	return report, root, nil
}
