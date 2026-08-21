package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_seal.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport

	AdmissionSealState                                        string `json:"admission_seal_state"`
	AdmissionSealAction                                       string `json:"admission_seal_action"`
	AdmissionSealTarget                                       string `json:"admission_seal_target"`
	AdmissionSealTargetKind                                   string `json:"admission_seal_target_kind"`
	AdmissionSealTargetMode                                   string `json:"admission_seal_target_mode"`
	AdmissionSealDryRunOnly                                   bool   `json:"admission_seal_dry_run_only"`
	AdmissionSealAuthorityVerified                            bool   `json:"admission_seal_authority_verified"`
	AdmissionSealPermitVerified                               bool   `json:"admission_seal_permit_verified"`
	AdmissionSealLedgerVerified                               bool   `json:"admission_seal_ledger_verified"`
	AdmissionSealReady                                        bool   `json:"admission_seal_ready"`
	AdmissionSealImmutableReceipt                             bool   `json:"admission_seal_immutable_receipt"`
	WeightedAdmissionResonanceGraftAdmissionSealReady         bool   `json:"weighted_admission_resonance_graft_admission_seal_ready"`
	WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed bool   `json:"weighted_admission_resonance_graft_admission_authority_consumed"`
	WeightedAdmissionResonanceGraftAdmissionAuthorityRequired bool   `json:"weighted_admission_resonance_graft_admission_authority_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionSeal         bool   `json:"next_step_blocked_without_resonance_graft_admission_seal"`
	WeightedAdmissionResonanceGraftAdmissionSealID            string `json:"weighted_admission_resonance_graft_admission_seal_id"`
	AdmissionSealHash                                         string `json:"admission_seal_hash"`
	AdmissionSealReadBackHash                                 string `json:"admission_seal_read_back_hash"`

	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID       string `json:"source_weighted_admission_resonance_graft_admission_authority_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady    bool   `json:"source_weighted_admission_resonance_graft_admission_authority_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID string `json:"source_weighted_admission_resonance_graft_admission_authority_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash     string `json:"source_weighted_admission_resonance_graft_admission_authority_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack string `json:"source_weighted_admission_resonance_graft_admission_authority_read_back_hash"`
	SourceAdmissionAuthorityReportReceiptShape                      string `json:"source_admission_authority_report_receipt_shape"`
	SourceAdmissionAuthorityState                                   string `json:"source_admission_authority_state"`
	SourceAdmissionAuthorityAction                                  string `json:"source_admission_authority_action"`
	SourceAdmissionAuthorityTarget                                  string `json:"source_admission_authority_target"`
	SourceAdmissionAuthorityTargetKind                              string `json:"source_admission_authority_target_kind"`
	SourceAdmissionAuthorityTargetMode                              string `json:"source_admission_authority_target_mode"`
	SourceAdmissionAuthorityDryRunOnly                              bool   `json:"source_admission_authority_dry_run_only"`
	SourceAdmissionAuthorityPermitVerified                          bool   `json:"source_admission_authority_permit_verified"`
	SourceAdmissionAuthorityLedgerVerified                          bool   `json:"source_admission_authority_ledger_verified"`
	SourceAdmissionAuthorityWriterReady                             bool   `json:"source_admission_authority_writer_ready"`
	SourceAdmissionAuthorityRollbackReady                           bool   `json:"source_admission_authority_rollback_ready"`
	SourceAdmissionAuthorityReady                                   bool   `json:"source_admission_authority_ready"`
	SourceAdmissionAuthorityGranted                                 bool   `json:"source_admission_authority_granted"`
	SourceAdmissionAuthorityReason                                  string `json:"source_admission_authority_reason"`
	SourceAdmissionPermitSchema                                     string `json:"source_admission_permit_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSeal(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-seal RESONANCE_GRAFT_ADMISSION_AUTHORITY_REPORT RESONANCE_GRAFT_ADMISSION_SEAL_REPORT")
	}
	authorityPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission seal output path missing")
	}
	sourceAuthority, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportForAssert(authorityPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReportError(sourceAuthority, root); err != nil {
		return err
	}
	seal := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthorityReport: sourceAuthority,
		AdmissionSealState:                                              "sealed",
		AdmissionSealAction:                                             "seal_blocked_admission_authority",
		AdmissionSealTarget:                                             "live_admission_authority",
		AdmissionSealTargetKind:                                         "weighted_internal_world_shadow_graft_admission_authority",
		AdmissionSealTargetMode:                                         "closed_seal_guard_dry_run",
		AdmissionSealDryRunOnly:                                         true,
		AdmissionSealAuthorityVerified:                                  false,
		AdmissionSealPermitVerified:                                     false,
		AdmissionSealLedgerVerified:                                     false,
		AdmissionSealReady:                                              false,
		AdmissionSealImmutableReceipt:                                   true,
		WeightedAdmissionResonanceGraftAdmissionSealReady:               true,
		WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionAuthorityRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionSeal:               true,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID:       sourceAuthority.WeightedAdmissionResonanceGraftAdmissionAuthorityID,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady:    sourceAuthority.WeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID: sourceAuthority.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash:     sourceAuthority.AdmissionAuthorityHash,
		SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack: sourceAuthority.AdmissionAuthorityReadBackHash,
		SourceAdmissionAuthorityReportReceiptShape:                      sourceAuthority.ReceiptShape,
		SourceAdmissionAuthorityState:                                   sourceAuthority.AdmissionAuthorityState,
		SourceAdmissionAuthorityAction:                                  sourceAuthority.AdmissionAuthorityAction,
		SourceAdmissionAuthorityTarget:                                  sourceAuthority.AdmissionAuthorityTarget,
		SourceAdmissionAuthorityTargetKind:                              sourceAuthority.AdmissionAuthorityTargetKind,
		SourceAdmissionAuthorityTargetMode:                              sourceAuthority.AdmissionAuthorityTargetMode,
		SourceAdmissionAuthorityDryRunOnly:                              sourceAuthority.AdmissionAuthorityDryRunOnly,
		SourceAdmissionAuthorityPermitVerified:                          sourceAuthority.AdmissionAuthorityPermitVerified,
		SourceAdmissionAuthorityLedgerVerified:                          sourceAuthority.AdmissionAuthorityLedgerVerified,
		SourceAdmissionAuthorityWriterReady:                             sourceAuthority.AdmissionAuthorityWriterReady,
		SourceAdmissionAuthorityRollbackReady:                           sourceAuthority.AdmissionAuthorityRollbackReady,
		SourceAdmissionAuthorityReady:                                   sourceAuthority.AdmissionAuthorityReady,
		SourceAdmissionAuthorityGranted:                                 sourceAuthority.AdmissionAuthorityGranted,
		SourceAdmissionAuthorityReason:                                  sourceAuthority.Reason,
		SourceAdmissionPermitSchema:                                     sourceAuthority.SourceSchema,
	}
	seal.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema
	seal.Status = "shadow_graft_admission_seal_blocked_dry_run"
	seal.TargetKind = "weighted_internal_world_shadow_graft_admission_seal"
	seal.TargetMode = "closed_seal_guard_dry_run"
	seal.Action = "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run"
	seal.WriterAction = "reject_blocked_admission_authority"
	seal.RollbackAction = "reject_blocked_admission_authority"
	seal.LedgerState = "blocked"
	seal.LedgerAction = "reject_blocked_admission_authority"
	seal.LedgerContract = "none"
	seal.LedgerEntrypoint = "none"
	seal.LedgerReceiptShape = "none"
	seal.LedgerWriteScope = "none"
	seal.LedgerReady = false
	seal.LedgerAppendAllowed = false
	seal.ReceiptShape = "weighted_resonance_shadow_graft_admission_seal_receipt"
	seal.SourceSchema = sourceAuthority.Schema
	seal.SourceStatus = sourceAuthority.Status
	seal.SourceTarget = sourceAuthority.Target
	seal.SourceReport = authorityPath
	seal.AuthorityGranted = false
	seal.Reason = "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed"
	seal.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealCausalID(seal)
	seal.AdmissionSealHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealHash(seal)
	seal.AdmissionSealReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReadBackHash(seal)
	seal.WeightedAdmissionResonanceGraftAdmissionSealID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealID(seal)
	if seal.CausalID == "" ||
		seal.AdmissionSealHash == "" ||
		seal.AdmissionSealReadBackHash == "" ||
		seal.WeightedAdmissionResonanceGraftAdmissionSealID == "" ||
		seal.AdmissionSealHash == seal.AdmissionSealReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission seal read-back proof failed")
	}
	raw, err := json.MarshalIndent(seal, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission seal marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission seal write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-seal] pass: resonance_graft_admission_seal_report=%s resonance_graft_admission_authority_report=%s\n", outputPath, authorityPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-seal-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission seal schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema {
		return fmt.Errorf("weighted admission resonance graft admission seal schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealSchema)
	}
	if report.Status != "shadow_graft_admission_seal_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal status mismatch: got %q want %q", report.Status, "shadow_graft_admission_seal_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission seal target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_seal" {
		return fmt.Errorf("weighted admission resonance graft admission seal target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_seal")
	}
	if report.TargetMode != "closed_seal_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal target_mode mismatch: got %q want %q", report.TargetMode, "closed_seal_guard_dry_run")
	}
	if report.Action != "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal action mismatch: got %q want %q", report.Action, "seal_weighted_resonance_shadow_graft_admission_authority_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_authority" || report.RollbackAction != "reject_blocked_admission_authority" {
		return fmt.Errorf("weighted admission resonance graft admission seal writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_authority" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission seal ledger guard mismatch")
	}
	if report.AdmissionSealState != "sealed" ||
		report.AdmissionSealAction != "seal_blocked_admission_authority" ||
		report.AdmissionSealTarget != "live_admission_authority" ||
		report.AdmissionSealTargetKind != "weighted_internal_world_shadow_graft_admission_authority" ||
		report.AdmissionSealTargetMode != "closed_seal_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_seal_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission seal receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_seal_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"admission_seal_immutable_receipt", report.AdmissionSealImmutableReceipt},
		{"weighted_admission_resonance_graft_admission_seal_ready", report.WeightedAdmissionResonanceGraftAdmissionSealReady},
		{"weighted_admission_resonance_graft_admission_authority_consumed", report.WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed},
		{"weighted_admission_resonance_graft_admission_authority_required", report.WeightedAdmissionResonanceGraftAdmissionAuthorityRequired},
		{"next_step_blocked_without_resonance_graft_admission_seal", report.NextStepBlockedWithoutResonanceGraftAdmissionSeal},
		{"source_weighted_admission_resonance_graft_admission_authority_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady},
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
			return fmt.Errorf("weighted admission resonance graft admission seal %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
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
			return fmt.Errorf("weighted admission resonance graft admission seal opened %s", closed.name)
		}
	}
	if !report.AdmissionSealDryRunOnly || !report.AdmissionAuthorityDryRunOnly || !report.SourceAdmissionAuthorityDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission seal dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_seal_id", report.WeightedAdmissionResonanceGraftAdmissionSealID},
		{"causal_id", report.CausalID},
		{"admission_seal_hash", report.AdmissionSealHash},
		{"admission_seal_read_back_hash", report.AdmissionSealReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_authority_id", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID},
		{"source_weighted_admission_resonance_graft_admission_authority_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID},
		{"source_weighted_admission_resonance_graft_admission_authority_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash},
		{"source_weighted_admission_resonance_graft_admission_authority_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack},
		{"source_admission_authority_reason", report.SourceAdmissionAuthorityReason},
		{"source_admission_permit_schema", report.SourceAdmissionPermitSchema},
		{"source_weighted_admission_resonance_graft_admission_permit_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID},
		{"source_weighted_admission_resonance_graft_admission_readiness_id", report.SourceWeightedAdmissionResonanceGraftAdmissionReadinessID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission seal %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema {
		return fmt.Errorf("weighted admission resonance graft admission seal source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionAuthoritySchema)
	}
	if report.SourceStatus != "shadow_graft_admission_authority_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_authority_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission seal source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceAdmissionPermitSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema {
		return fmt.Errorf("weighted admission resonance graft admission seal source_admission_permit_schema mismatch: got %q want %q", report.SourceAdmissionPermitSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPermitSchema)
	}
	if report.SourceAdmissionAuthorityReportReceiptShape != "weighted_resonance_shadow_graft_admission_authority_receipt" ||
		report.SourceAdmissionAuthorityState != "blocked" ||
		report.SourceAdmissionAuthorityAction != "reject_blocked_admission_permit" ||
		report.SourceAdmissionAuthorityTarget != "live_admission_authority" ||
		report.SourceAdmissionAuthorityTargetKind != "weighted_internal_world_shadow_graft_admission_permit" ||
		report.SourceAdmissionAuthorityTargetMode != "closed_authority_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission seal source admission authority shape mismatch")
	}
	if report.SourceAdmissionAuthorityReason != "weighted resonance shadow graft admission authority blocked by blocked permit; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission seal source_admission_authority_reason mismatch: got %q", report.SourceAdmissionAuthorityReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID, "weighted-resonance-graft-admission-authority-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityCausalID, "weighted-resonance-graft-admission-authority-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash, "weighted-resonance-graft-admission-authority-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack, "weighted-resonance-graft-admission-authority-read-") {
		return fmt.Errorf("weighted admission resonance graft admission seal source authority mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission seal body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission seal causal_id mismatch")
	}
	if report.AdmissionSealHash == "" || report.AdmissionSealHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission seal admission_seal_hash mismatch")
	}
	if report.AdmissionSealReadBackHash == "" || report.AdmissionSealReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission seal admission_seal_read_back_hash mismatch")
	}
	if report.AdmissionSealHash == report.AdmissionSealReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission seal read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionSealID == "" || report.WeightedAdmissionResonanceGraftAdmissionSealID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealID(report) {
		return fmt.Errorf("weighted admission resonance graft admission seal id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission seal fixed blocked authority provenance; live authority remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission seal reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport) string {
	h := hashJSON(struct {
		SourceAuthorityID   string `json:"source_admission_authority_id"`
		SourceAuthorityRead string `json:"source_admission_authority_read_back_hash"`
		SourcePermitID      string `json:"source_admission_permit_id"`
		Target              string `json:"target"`
		State               string `json:"admission_seal_state"`
		Action              string `json:"admission_seal_action"`
	}{
		SourceAuthorityID:   report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID,
		SourceAuthorityRead: report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack,
		SourcePermitID:      report.SourceWeightedAdmissionResonanceGraftAdmissionPermitID,
		Target:              report.Target,
		State:               report.AdmissionSealState,
		Action:              report.AdmissionSealAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-seal-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourceAuthorityID      string `json:"source_admission_authority_id"`
		SourceAuthorityHash    string `json:"source_admission_authority_hash"`
		SourceAuthorityRead    string `json:"source_admission_authority_read_back_hash"`
		State                  string `json:"admission_seal_state"`
		Action                 string `json:"admission_seal_action"`
		Target                 string `json:"admission_seal_target"`
		TargetKind             string `json:"admission_seal_target_kind"`
		TargetMode             string `json:"admission_seal_target_mode"`
		DryRunOnly             bool   `json:"admission_seal_dry_run_only"`
		AuthorityVerified      bool   `json:"admission_seal_authority_verified"`
		PermitVerified         bool   `json:"admission_seal_permit_verified"`
		LedgerVerified         bool   `json:"admission_seal_ledger_verified"`
		Ready                  bool   `json:"admission_seal_ready"`
		ImmutableReceipt       bool   `json:"admission_seal_immutable_receipt"`
		WeightedReady          bool   `json:"weighted_seal_ready"`
		SourceWeightedReady    bool   `json:"source_weighted_authority_ready"`
		SourceAuthorityReady   bool   `json:"source_admission_authority_ready"`
		SourceAuthorityGranted bool   `json:"source_admission_authority_granted"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_seal"`
	}{
		CausalID:               report.CausalID,
		SourceAuthorityID:      report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID,
		SourceAuthorityHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash,
		SourceAuthorityRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack,
		State:                  report.AdmissionSealState,
		Action:                 report.AdmissionSealAction,
		Target:                 report.AdmissionSealTarget,
		TargetKind:             report.AdmissionSealTargetKind,
		TargetMode:             report.AdmissionSealTargetMode,
		DryRunOnly:             report.AdmissionSealDryRunOnly,
		AuthorityVerified:      report.AdmissionSealAuthorityVerified,
		PermitVerified:         report.AdmissionSealPermitVerified,
		LedgerVerified:         report.AdmissionSealLedgerVerified,
		Ready:                  report.AdmissionSealReady,
		ImmutableReceipt:       report.AdmissionSealImmutableReceipt,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionSealReady,
		SourceWeightedReady:    report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReady,
		SourceAuthorityReady:   report.SourceAdmissionAuthorityReady,
		SourceAuthorityGranted: report.SourceAdmissionAuthorityGranted,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionSeal,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-seal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport) string {
	h := hashJSON(struct {
		AdmissionSealHash   string `json:"admission_seal_hash"`
		SourceAuthorityID   string `json:"source_admission_authority_id"`
		SourceAuthorityRead string `json:"source_admission_authority_read_back_hash"`
		WeightedReady       bool   `json:"weighted_seal_ready"`
		AuthorityConsumed   bool   `json:"authority_consumed"`
		AuthorityRequired   bool   `json:"authority_required"`
		SealReady           bool   `json:"admission_seal_ready"`
		ImmutableReceipt    bool   `json:"admission_seal_immutable_receipt"`
		AuthorityVerified   bool   `json:"admission_seal_authority_verified"`
		WriteAllowed        bool   `json:"write_allowed"`
		AdmissionAllowed    bool   `json:"admission_allowed"`
		LiveEnabled         bool   `json:"live_admission_enabled"`
		MutatesState        bool   `json:"mutates_state"`
	}{
		AdmissionSealHash:   report.AdmissionSealHash,
		SourceAuthorityID:   report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID,
		SourceAuthorityRead: report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack,
		WeightedReady:       report.WeightedAdmissionResonanceGraftAdmissionSealReady,
		AuthorityConsumed:   report.WeightedAdmissionResonanceGraftAdmissionAuthorityConsumed,
		AuthorityRequired:   report.WeightedAdmissionResonanceGraftAdmissionAuthorityRequired,
		SealReady:           report.AdmissionSealReady,
		ImmutableReceipt:    report.AdmissionSealImmutableReceipt,
		AuthorityVerified:   report.AdmissionSealAuthorityVerified,
		WriteAllowed:        report.WriteAllowed,
		AdmissionAllowed:    report.AdmissionAllowed,
		LiveEnabled:         report.LiveAdmissionEnabled,
		MutatesState:        report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-seal-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceAuthorityID      string `json:"source_admission_authority_id"`
		SourceAuthorityHash    string `json:"source_admission_authority_hash"`
		SourceAuthorityRead    string `json:"source_admission_authority_read_back_hash"`
		CausalID               string `json:"causal_id"`
		AdmissionSealHash      string `json:"admission_seal_hash"`
		AdmissionSealRead      string `json:"admission_seal_read_back_hash"`
		State                  string `json:"admission_seal_state"`
		ActionSeal             string `json:"admission_seal_action"`
		Ready                  bool   `json:"weighted_seal_ready"`
		SealReady              bool   `json:"admission_seal_ready"`
		ImmutableReceipt       bool   `json:"admission_seal_immutable_receipt"`
		AuthorityVerified      bool   `json:"admission_seal_authority_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_seal"`
	}{
		Schema:                 report.Schema,
		Status:                 report.Status,
		Action:                 report.Action,
		SourceAuthorityID:      report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityID,
		SourceAuthorityHash:    report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityHash,
		SourceAuthorityRead:    report.SourceWeightedAdmissionResonanceGraftAdmissionAuthorityReadBack,
		CausalID:               report.CausalID,
		AdmissionSealHash:      report.AdmissionSealHash,
		AdmissionSealRead:      report.AdmissionSealReadBackHash,
		State:                  report.AdmissionSealState,
		ActionSeal:             report.AdmissionSealAction,
		Ready:                  report.WeightedAdmissionResonanceGraftAdmissionSealReady,
		SealReady:              report.AdmissionSealReady,
		ImmutableReceipt:       report.AdmissionSealImmutableReceipt,
		AuthorityVerified:      report.AdmissionSealAuthorityVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionSeal,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-seal-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSealReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission seal path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission seal not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission seal not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission seal JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission seal decode failed: %w", err)
	}
	return report, root, nil
}
