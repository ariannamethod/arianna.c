package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema = "arianna.live_route_weighted_admission_resonance_graft_admission_authority.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport

	AdmissionAuthorityState                                string `json:"admission_authority_state"`
	AdmissionAuthorityAction                               string `json:"admission_authority_action"`
	AdmissionAuthorityTarget                               string `json:"admission_authority_target"`
	AdmissionAuthorityTargetKind                           string `json:"admission_authority_target_kind"`
	AdmissionAuthorityTargetMode                           string `json:"admission_authority_target_mode"`
	AdmissionAuthorityDryRunOnly                           bool   `json:"admission_authority_dry_run_only"`
	AdmissionAuthorityPermitVerified                       bool   `json:"admission_authority_permit_verified"`
	AdmissionAuthorityLedgerVerified                       bool   `json:"admission_authority_ledger_verified"`
	AdmissionAuthorityWriterReady                          bool   `json:"admission_authority_writer_ready"`
	AdmissionAuthorityRollbackReady                        bool   `json:"admission_authority_rollback_ready"`
	AdmissionAuthorityReady                                bool   `json:"admission_authority_ready"`
	AdmissionAuthorityGranted                              bool   `json:"admission_authority_granted"`
	WeightedAdmissionResonanceGraftAdmissionAuthorityReady bool   `json:"weighted_admission_resonance_graft_admission_authority_ready"`
	WeightedAdmissionResonanceGraftAdmissionPermitConsumed bool   `json:"weighted_admission_resonance_graft_admission_permit_consumed"`
	WeightedAdmissionResonanceGraftAdmissionPermitRequired bool   `json:"weighted_admission_resonance_graft_admission_permit_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionAuthority bool   `json:"next_step_blocked_without_resonance_graft_admission_authority"`
	WeightedAdmissionResonanceGraftAdmissionAuthorityID    string `json:"weighted_admission_resonance_graft_admission_authority_id"`
	AdmissionAuthorityHash                                 string `json:"admission_authority_hash"`
	AdmissionAuthorityReadBackHash                         string `json:"admission_authority_read_back_hash"`

	SourceWeightedAdmissionResonanceGraftAdmissionPermitID       string `json:"source_weighted_admission_resonance_graft_admission_permit_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReady    bool   `json:"source_weighted_admission_resonance_graft_admission_permit_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitCausalID string `json:"source_weighted_admission_resonance_graft_admission_permit_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitHash     string `json:"source_weighted_admission_resonance_graft_admission_permit_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack string `json:"source_weighted_admission_resonance_graft_admission_permit_read_back_hash"`
	SourceAdmissionPermitReportReceiptShape                      string `json:"source_admission_permit_report_receipt_shape"`
	SourceAdmissionPermitState                                   string `json:"source_admission_permit_state"`
	SourceAdmissionPermitAction                                  string `json:"source_admission_permit_action"`
	SourceAdmissionPermitTarget                                  string `json:"source_admission_permit_target"`
	SourceAdmissionPermitTargetKind                              string `json:"source_admission_permit_target_kind"`
	SourceAdmissionPermitTargetMode                              string `json:"source_admission_permit_target_mode"`
	SourceAdmissionPermitDryRunOnly                              bool   `json:"source_admission_permit_dry_run_only"`
	SourceAdmissionPermitReadinessVerified                       bool   `json:"source_admission_permit_readiness_verified"`
	SourceAdmissionPermitLedgerVerified                          bool   `json:"source_admission_permit_ledger_verified"`
	SourceAdmissionPermitWriterReady                             bool   `json:"source_admission_permit_writer_ready"`
	SourceAdmissionPermitRollbackReady                           bool   `json:"source_admission_permit_rollback_ready"`
	SourceAdmissionPermitLedgerReady                             bool   `json:"source_admission_permit_ledger_ready"`
	SourceAdmissionPermitReady                                   bool   `json:"source_admission_permit_ready"`
	SourceManualPermitRequested                                  bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                                       bool   `json:"source_permit_key_matched"`
	SourceAdmissionPermitReason                                  string `json:"source_admission_permit_reason"`
	SourceAdmissionReadinessSchema                               string `json:"source_admission_readiness_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthority(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-authority RESONANCE_GRAFT_ADMISSION_PERMIT_REPORT RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT")
	}
	permitPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission authority output path missing")
	}
	sourcePermit, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportForAssert(permitPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReportError(sourcePermit, root); err != nil {
		return err
	}
	authority := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitReport: sourcePermit,
		AdmissionAuthorityState:                                      "blocked",
		AdmissionAuthorityAction:                                     "reject_blocked_admission_permit",
		AdmissionAuthorityTarget:                                     "live_admission_authority",
		AdmissionAuthorityTargetKind:                                 "weighted_internal_world_shadow_graft_admission_permit",
		AdmissionAuthorityTargetMode:                                 "closed_authority_guard_dry_run",
		AdmissionAuthorityDryRunOnly:                                 true,
		AdmissionAuthorityPermitVerified:                             false,
		AdmissionAuthorityLedgerVerified:                             false,
		AdmissionAuthorityWriterReady:                                false,
		AdmissionAuthorityRollbackReady:                              false,
		AdmissionAuthorityReady:                                      false,
		AdmissionAuthorityGranted:                                    false,
		WeightedAdmissionResonanceGraftAdmissionAuthorityReady:       true,
		WeightedAdmissionResonanceGraftAdmissionPermitConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionPermitRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionAuthority:       true,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitID:       sourcePermit.WeightedAdmissionResonanceGraftAdmissionPermitID,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReady:    sourcePermit.WeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitCausalID: sourcePermit.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitHash:     sourcePermit.AdmissionPermitHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack: sourcePermit.AdmissionPermitReadBackHash,
		SourceAdmissionPermitReportReceiptShape:                      sourcePermit.ReceiptShape,
		SourceAdmissionPermitState:                                   sourcePermit.AdmissionPermitState,
		SourceAdmissionPermitAction:                                  sourcePermit.AdmissionPermitAction,
		SourceAdmissionPermitTarget:                                  sourcePermit.AdmissionPermitTarget,
		SourceAdmissionPermitTargetKind:                              sourcePermit.AdmissionPermitTargetKind,
		SourceAdmissionPermitTargetMode:                              sourcePermit.AdmissionPermitTargetMode,
		SourceAdmissionPermitDryRunOnly:                              sourcePermit.AdmissionPermitDryRunOnly,
		SourceAdmissionPermitReadinessVerified:                       sourcePermit.AdmissionPermitReadinessVerified,
		SourceAdmissionPermitLedgerVerified:                          sourcePermit.AdmissionPermitLedgerVerified,
		SourceAdmissionPermitWriterReady:                             sourcePermit.AdmissionPermitWriterReady,
		SourceAdmissionPermitRollbackReady:                           sourcePermit.AdmissionPermitRollbackReady,
		SourceAdmissionPermitLedgerReady:                             sourcePermit.AdmissionPermitLedgerReady,
		SourceAdmissionPermitReady:                                   sourcePermit.AdmissionPermitReady,
		SourceManualPermitRequested:                                  sourcePermit.ManualPermitRequested,
		SourcePermitKeyMatched:                                       sourcePermit.PermitKeyMatched,
		SourceAdmissionPermitReason:                                  sourcePermit.Reason,
		SourceAdmissionReadinessSchema:                               sourcePermit.SourceSchema,
	}
	authority.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema
	authority.Status = "shadow_graft_admission_authority_blocked_dry_run"
	authority.TargetKind = "weighted_internal_world_shadow_graft_admission_authority"
	authority.TargetMode = "closed_authority_guard_dry_run"
	authority.Action = "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run"
	authority.WriterAction = "reject_blocked_admission_permit"
	authority.RollbackAction = "reject_blocked_admission_permit"
	authority.LedgerState = "blocked"
	authority.LedgerAction = "reject_blocked_admission_permit"
	authority.LedgerContract = "none"
	authority.LedgerEntrypoint = "none"
	authority.LedgerReceiptShape = "none"
	authority.LedgerWriteScope = "none"
	authority.LedgerReady = false
	authority.LedgerAppendAllowed = false
	authority.ReceiptShape = "weighted_resonance_shadow_graft_admission_authority_receipt"
	authority.SourceSchema = sourcePermit.Schema
	authority.SourceStatus = sourcePermit.Status
	authority.SourceTarget = sourcePermit.Target
	authority.SourceReport = permitPath
	authority.AuthorityGranted = false
	authority.Reason = "weighted resonance shadow graft admission authority blocked by blocked permit; live authority remains closed"
	authority.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID(authority)
	authority.AdmissionAuthorityHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityHash(authority)
	authority.AdmissionAuthorityReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReadBackHash(authority)
	authority.WeightedAdmissionResonanceGraftAdmissionAuthorityID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityID(authority)
	if authority.CausalID == "" ||
		authority.AdmissionAuthorityHash == "" ||
		authority.AdmissionAuthorityReadBackHash == "" ||
		authority.WeightedAdmissionResonanceGraftAdmissionAuthorityID == "" ||
		authority.AdmissionAuthorityHash == authority.AdmissionAuthorityReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission authority read-back proof failed")
	}
	raw, err := json.MarshalIndent(authority, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission authority marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission authority write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-authority] pass: resonance_graft_admission_authority_report=%s resonance_graft_admission_permit_report=%s\n", outputPath, permitPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-authority-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission authority schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema {
		return fmt.Errorf("weighted admission resonance graft admission authority schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema)
	}
	if report.Status != "shadow_graft_admission_authority_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority status mismatch: got %q want %q", report.Status, "shadow_graft_admission_authority_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission authority target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_authority" {
		return fmt.Errorf("weighted admission resonance graft admission authority target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_authority")
	}
	if report.TargetMode != "closed_authority_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority target_mode mismatch: got %q want %q", report.TargetMode, "closed_authority_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_permit_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_permit" || report.RollbackAction != "reject_blocked_admission_permit" {
		return fmt.Errorf("weighted admission resonance graft admission authority writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_permit" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission authority ledger guard mismatch")
	}
	if report.AdmissionAuthorityState != "blocked" ||
		report.AdmissionAuthorityAction != "reject_blocked_admission_permit" ||
		report.AdmissionAuthorityTarget != "live_admission_authority" ||
		report.AdmissionAuthorityTargetKind != "weighted_internal_world_shadow_graft_admission_permit" ||
		report.AdmissionAuthorityTargetMode != "closed_authority_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_authority_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission authority receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_authority_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_authority_ready", report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady},
		{"weighted_admission_resonance_graft_admission_permit_consumed", report.WeightedAdmissionResonanceGraftAdmissionPermitConsumed},
		{"weighted_admission_resonance_graft_admission_permit_required", report.WeightedAdmissionResonanceGraftAdmissionPermitRequired},
		{"next_step_blocked_without_resonance_graft_admission_authority", report.NextStepBlockedWithoutResonanceGraftAdmissionAuthority},
		{"source_weighted_admission_resonance_graft_admission_permit_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"weighted_admission_resonance_graft_admission_permit_ready", report.WeightedAdmissionResonanceGraftAdmissionPermitReady},
		{"weighted_admission_resonance_graft_admission_readiness_consumed", report.WeightedAdmissionResonanceGraftAdmissionReadinessConsumed},
		{"weighted_admission_resonance_graft_admission_readiness_required", report.WeightedAdmissionResonanceGraftAdmissionReadinessRequired},
		{"weighted_admission_resonance_graft_admission_readiness_ready", report.WeightedAdmissionResonanceGraftAdmissionReadinessReady},
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
			return fmt.Errorf("weighted admission resonance graft admission authority %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"admission_authority_permit_verified", report.AdmissionAuthorityPermitVerified},
		{"admission_authority_ledger_verified", report.AdmissionAuthorityLedgerVerified},
		{"admission_authority_writer_ready", report.AdmissionAuthorityWriterReady},
		{"admission_authority_rollback_ready", report.AdmissionAuthorityRollbackReady},
		{"admission_authority_ready", report.AdmissionAuthorityReady},
		{"admission_authority_granted", report.AdmissionAuthorityGranted},
		{"source_admission_permit_readiness_verified", report.SourceAdmissionPermitReadinessVerified},
		{"source_admission_permit_ledger_verified", report.SourceAdmissionPermitLedgerVerified},
		{"source_admission_permit_writer_ready", report.SourceAdmissionPermitWriterReady},
		{"source_admission_permit_rollback_ready", report.SourceAdmissionPermitRollbackReady},
		{"source_admission_permit_ledger_ready", report.SourceAdmissionPermitLedgerReady},
		{"source_admission_permit_ready", report.SourceAdmissionPermitReady},
		{"source_manual_permit_requested", report.SourceManualPermitRequested},
		{"source_permit_key_matched", report.SourcePermitKeyMatched},
		{"manual_permit_requested", report.ManualPermitRequested},
		{"permit_key_matched", report.PermitKeyMatched},
		{"admission_permit_readiness_verified", report.AdmissionPermitReadinessVerified},
		{"admission_permit_ledger_verified", report.AdmissionPermitLedgerVerified},
		{"admission_permit_writer_ready", report.AdmissionPermitWriterReady},
		{"admission_permit_rollback_ready", report.AdmissionPermitRollbackReady},
		{"admission_permit_ledger_ready", report.AdmissionPermitLedgerReady},
		{"admission_permit_ready", report.AdmissionPermitReady},
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
			return fmt.Errorf("weighted admission resonance graft admission authority opened %s", closed.name)
		}
	}
	if !report.AdmissionAuthorityDryRunOnly || !report.AdmissionPermitDryRunOnly || !report.SourceAdmissionPermitDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission authority dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_authority_id", report.WeightedAdmissionResonanceGraftAdmissionAuthorityID},
		{"causal_id", report.CausalID},
		{"admission_authority_hash", report.AdmissionAuthorityHash},
		{"admission_authority_read_back_hash", report.AdmissionAuthorityReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_permit_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID},
		{"source_weighted_admission_resonance_graft_admission_permit_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitCausalID},
		{"source_weighted_admission_resonance_graft_admission_permit_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitHash},
		{"source_weighted_admission_resonance_graft_admission_permit_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack},
		{"source_admission_permit_reason", report.SourceAdmissionPermitReason},
		{"source_admission_readiness_schema", report.SourceAdmissionReadinessSchema},
		{"source_weighted_admission_resonance_graft_admission_readiness_id", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission authority %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema {
		return fmt.Errorf("weighted admission resonance graft admission authority source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_permit_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_permit_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission authority source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionReadinessSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema {
		return fmt.Errorf("weighted admission resonance graft admission authority source_admission_readiness_schema mismatch: got %q want %q", report.SourceAdmissionReadinessSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema)
	}
	if report.SourceAdmissionPermitReportReceiptShape != "weighted_resonance_shadow_graft_admission_permit_receipt" ||
		report.SourceAdmissionPermitState != "blocked" ||
		report.SourceAdmissionPermitAction != "reject_blocked_admission_readiness" ||
		report.SourceAdmissionPermitTarget != "live_admission" ||
		report.SourceAdmissionPermitTargetKind != "weighted_internal_world_shadow_graft_admission_readiness" ||
		report.SourceAdmissionPermitTargetMode != "closed_permit_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission authority source admission permit shape mismatch")
	}
	if report.SourceAdmissionPermitReason != "weighted resonance shadow graft admission permit blocked by blocked readiness; manual permit remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission authority source_admission_permit_reason mismatch: got %q", report.SourceAdmissionPermitReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID, "weighted-resonance-graft-admission-permit-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPermitCausalID, "weighted-resonance-graft-admission-permit-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPermitHash, "weighted-resonance-graft-admission-permit-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack, "weighted-resonance-graft-admission-permit-read-") {
		return fmt.Errorf("weighted admission resonance graft admission authority source permit mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission authority body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission authority causal_id mismatch")
	}
	if report.AdmissionAuthorityHash == "" || report.AdmissionAuthorityHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission authority admission_authority_hash mismatch")
	}
	if report.AdmissionAuthorityReadBackHash == "" || report.AdmissionAuthorityReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission authority admission_authority_read_back_hash mismatch")
	}
	if report.AdmissionAuthorityHash == report.AdmissionAuthorityReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission authority read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionAuthorityID == "" || report.WeightedAdmissionResonanceGraftAdmissionAuthorityID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityID(report) {
		return fmt.Errorf("weighted admission resonance graft admission authority id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission authority blocked by blocked permit; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission authority reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport) string {
	h := hashJSON(struct {
		SourcePermitID   string `json:"source_admission_permit_id"`
		SourcePermitRead string `json:"source_admission_permit_read_back_hash"`
		SourceReadiness  string `json:"source_admission_readiness_id"`
		Target           string `json:"target"`
		State            string `json:"admission_authority_state"`
		Action           string `json:"admission_authority_action"`
	}{
		SourcePermitID:   report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID,
		SourcePermitRead: report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack,
		SourceReadiness:  report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID,
		Target:           report.Target,
		State:            report.AdmissionAuthorityState,
		Action:           report.AdmissionAuthorityAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-authority-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourcePermitID         string `json:"source_admission_permit_id"`
		SourcePermitHash       string `json:"source_admission_permit_hash"`
		SourcePermitRead       string `json:"source_admission_permit_read_back_hash"`
		State                  string `json:"admission_authority_state"`
		Action                 string `json:"admission_authority_action"`
		Target                 string `json:"admission_authority_target"`
		TargetKind             string `json:"admission_authority_target_kind"`
		TargetMode             string `json:"admission_authority_target_mode"`
		DryRunOnly             bool   `json:"admission_authority_dry_run_only"`
		PermitVerified         bool   `json:"admission_authority_permit_verified"`
		LedgerVerified         bool   `json:"admission_authority_ledger_verified"`
		WriterReady            bool   `json:"admission_authority_writer_ready"`
		RollbackReady          bool   `json:"admission_authority_rollback_ready"`
		Ready                  bool   `json:"admission_authority_ready"`
		Granted                bool   `json:"admission_authority_granted"`
		WeightedReady          bool   `json:"weighted_authority_ready"`
		SourceWeightedReady    bool   `json:"source_weighted_permit_ready"`
		SourcePermitReady      bool   `json:"source_admission_permit_ready"`
		ManualRequested        bool   `json:"manual_permit_requested"`
		KeyMatched             bool   `json:"permit_key_matched"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_authority"`
	}{
		CausalID:               report.CausalID,
		SourcePermitID:         report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID,
		SourcePermitHash:       report.SourceWeightedAdmissionResonanceGraftAdmissionPermitHash,
		SourcePermitRead:       report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack,
		State:                  report.AdmissionAuthorityState,
		Action:                 report.AdmissionAuthorityAction,
		Target:                 report.AdmissionAuthorityTarget,
		TargetKind:             report.AdmissionAuthorityTargetKind,
		TargetMode:             report.AdmissionAuthorityTargetMode,
		DryRunOnly:             report.AdmissionAuthorityDryRunOnly,
		PermitVerified:         report.AdmissionAuthorityPermitVerified,
		LedgerVerified:         report.AdmissionAuthorityLedgerVerified,
		WriterReady:            report.AdmissionAuthorityWriterReady,
		RollbackReady:          report.AdmissionAuthorityRollbackReady,
		Ready:                  report.AdmissionAuthorityReady,
		Granted:                report.AdmissionAuthorityGranted,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReady,
		SourcePermitReady:      report.SourceAdmissionPermitReady,
		ManualRequested:        report.ManualPermitRequested,
		KeyMatched:             report.PermitKeyMatched,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionAuthority,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-authority-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport) string {
	h := hashJSON(struct {
		AdmissionAuthorityHash string `json:"admission_authority_hash"`
		SourcePermitID         string `json:"source_admission_permit_id"`
		SourcePermitRead       string `json:"source_admission_permit_read_back_hash"`
		WeightedReady          bool   `json:"weighted_authority_ready"`
		PermitConsumed         bool   `json:"permit_consumed"`
		PermitRequired         bool   `json:"permit_required"`
		AuthorityReady         bool   `json:"admission_authority_ready"`
		AuthorityGranted       bool   `json:"admission_authority_granted"`
		PermitVerified         bool   `json:"admission_authority_permit_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveEnabled            bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		AdmissionAuthorityHash: report.AdmissionAuthorityHash,
		SourcePermitID:         report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID,
		SourcePermitRead:       report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		PermitConsumed:         report.WeightedAdmissionResonanceGraftAdmissionPermitConsumed,
		PermitRequired:         report.WeightedAdmissionResonanceGraftAdmissionPermitRequired,
		AuthorityReady:         report.AdmissionAuthorityReady,
		AuthorityGranted:       report.AdmissionAuthorityGranted,
		PermitVerified:         report.AdmissionAuthorityPermitVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveEnabled:            report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-authority-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourcePermitID         string `json:"source_admission_permit_id"`
		SourcePermitHash       string `json:"source_admission_permit_hash"`
		SourcePermitRead       string `json:"source_admission_permit_read_back_hash"`
		CausalID               string `json:"causal_id"`
		AdmissionAuthorityHash string `json:"admission_authority_hash"`
		AdmissionAuthorityRead string `json:"admission_authority_read_back_hash"`
		State                  string `json:"admission_authority_state"`
		ActionAuthority        string `json:"admission_authority_action"`
		Ready                  bool   `json:"weighted_authority_ready"`
		AuthorityReady         bool   `json:"admission_authority_ready"`
		AuthorityGranted       bool   `json:"admission_authority_granted"`
		PermitVerified         bool   `json:"admission_authority_permit_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_authority"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourcePermitID:         report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID,
		SourcePermitHash:       report.SourceWeightedAdmissionResonanceGraftAdmissionPermitHash,
		SourcePermitRead:       report.SourceWeightedAdmissionResonanceGraftAdmissionPermitReadBack,
		CausalID:               report.CausalID,
		AdmissionAuthorityHash: report.AdmissionAuthorityHash,
		AdmissionAuthorityRead: report.AdmissionAuthorityReadBackHash,
		State:                  report.AdmissionAuthorityState,
		ActionAuthority:        report.AdmissionAuthorityAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		AuthorityReady:         report.AdmissionAuthorityReady,
		AuthorityGranted:       report.AdmissionAuthorityGranted,
		PermitVerified:         report.AdmissionAuthorityPermitVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionAuthority,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-authority-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission authority path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission authority not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission authority not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission authority JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission authority decode failed: %w", err)
	}
	return report, root, nil
}
