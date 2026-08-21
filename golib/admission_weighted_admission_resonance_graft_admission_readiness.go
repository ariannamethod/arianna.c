package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_readiness.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport

	AdmissionReadinessState          string `json:"admission_readiness_state"`
	AdmissionReadinessAction         string `json:"admission_readiness_action"`
	AdmissionReadinessTarget         string `json:"admission_readiness_target"`
	AdmissionReadinessTargetKind     string `json:"admission_readiness_target_kind"`
	AdmissionReadinessTargetMode     string `json:"admission_readiness_target_mode"`
	AdmissionReadinessDryRunOnly     bool   `json:"admission_readiness_dry_run_only"`
	AdmissionReadinessLedgerVerified bool   `json:"admission_readiness_ledger_verified"`
	AdmissionReadinessWriterReady    bool   `json:"admission_readiness_writer_ready"`
	AdmissionReadinessRollbackReady  bool   `json:"admission_readiness_rollback_ready"`
	AdmissionReadinessLedgerReady    bool   `json:"admission_readiness_ledger_ready"`
	AdmissionReadinessReady          bool   `json:"admission_readiness_ready"`

	WeightedAdmissionResonanceGraftAdmissionReadinessReady             bool   `json:"weighted_admission_resonance_graft_admission_readiness_ready"`
	WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed bool   `json:"weighted_admission_resonance_graft_admission_ledger_verification_consumed"`
	WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired bool   `json:"weighted_admission_resonance_graft_admission_ledger_verification_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionReadiness             bool   `json:"next_step_blocked_without_resonance_graft_admission_readiness"`
	WeightedAdmissionResonanceGraftAdmissionReadinessID                string `json:"weighted_admission_resonance_graft_admission_readiness_id"`
	AdmissionReadinessHash                                             string `json:"admission_readiness_hash"`
	AdmissionReadinessReadBackHash                                     string `json:"admission_readiness_read_back_hash"`

	SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID       string `json:"source_weighted_admission_resonance_graft_admission_ledger_verification_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady    bool   `json:"source_weighted_admission_resonance_graft_admission_ledger_verification_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID string `json:"source_weighted_admission_resonance_graft_admission_ledger_verification_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash     string `json:"source_weighted_admission_resonance_graft_admission_ledger_verification_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack string `json:"source_weighted_admission_resonance_graft_admission_ledger_verification_read_back_hash"`
	SourceLedgerVerificationReportReceiptShape                               string `json:"source_ledger_verification_report_receipt_shape"`
	SourceLedgerVerificationState                                            string `json:"source_ledger_verification_state"`
	SourceLedgerVerificationAction                                           string `json:"source_ledger_verification_action"`
	SourceLedgerVerificationTarget                                           string `json:"source_ledger_verification_target"`
	SourceLedgerVerificationTargetKind                                       string `json:"source_ledger_verification_target_kind"`
	SourceLedgerVerificationTargetMode                                       string `json:"source_ledger_verification_target_mode"`
	SourceLedgerVerificationReceiptShape                                     string `json:"source_ledger_verification_receipt_shape"`
	SourceLedgerVerificationAppendOnly                                       bool   `json:"source_ledger_verification_append_only"`
	SourceLedgerVerificationDryRunOnly                                       bool   `json:"source_ledger_verification_dry_run_only"`
	SourceLedgerVerificationReceiptReadBack                                  bool   `json:"source_ledger_verification_receipt_read_back"`
	SourceLedgerVerificationReceiptVerified                                  bool   `json:"source_ledger_verification_receipt_verified"`
	SourceLedgerVerificationReady                                            bool   `json:"source_ledger_verification_ready"`
	SourceLedgerVerificationReason                                           string `json:"source_ledger_verification_reason"`
	SourceLedgerPersistenceSchema                                            string `json:"source_ledger_persistence_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadiness(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-readiness RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT RESONANCE_GRAFT_ADMISSION_READINESS_REPORT")
	}
	verificationPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission readiness output path missing")
	}
	sourceVerification, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportForAssert(verificationPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportError(sourceVerification, root); err != nil {
		return err
	}
	readiness := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport: sourceVerification,
		AdmissionReadinessState:                                                  "blocked",
		AdmissionReadinessAction:                                                 "reject_blocked_ledger_verification",
		AdmissionReadinessTarget:                                                 "live_admission",
		AdmissionReadinessTargetKind:                                             "weighted_internal_world_shadow_graft_admission_ledger_verification",
		AdmissionReadinessTargetMode:                                             "closed_readiness_guard_dry_run",
		AdmissionReadinessDryRunOnly:                                             true,
		AdmissionReadinessLedgerVerified:                                         false,
		AdmissionReadinessWriterReady:                                            false,
		AdmissionReadinessRollbackReady:                                          false,
		AdmissionReadinessLedgerReady:                                            false,
		AdmissionReadinessReady:                                                  false,
		WeightedAdmissionResonanceGraftAdmissionReadinessReady:                   true,
		WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionReadiness:                   true,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID:       sourceVerification.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady:    sourceVerification.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID: sourceVerification.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash:     sourceVerification.LedgerVerificationHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack: sourceVerification.LedgerVerificationReadBackHash,
		SourceLedgerVerificationReportReceiptShape:                               sourceVerification.ReceiptShape,
		SourceLedgerVerificationState:                                            sourceVerification.LedgerVerificationState,
		SourceLedgerVerificationAction:                                           sourceVerification.LedgerVerificationAction,
		SourceLedgerVerificationTarget:                                           sourceVerification.LedgerVerificationTarget,
		SourceLedgerVerificationTargetKind:                                       sourceVerification.LedgerVerificationTargetKind,
		SourceLedgerVerificationTargetMode:                                       sourceVerification.LedgerVerificationTargetMode,
		SourceLedgerVerificationReceiptShape:                                     sourceVerification.LedgerVerificationReceiptShape,
		SourceLedgerVerificationAppendOnly:                                       sourceVerification.LedgerVerificationAppendOnly,
		SourceLedgerVerificationDryRunOnly:                                       sourceVerification.LedgerVerificationDryRunOnly,
		SourceLedgerVerificationReceiptReadBack:                                  sourceVerification.LedgerVerificationReceiptReadBack,
		SourceLedgerVerificationReceiptVerified:                                  sourceVerification.LedgerVerificationReceiptVerified,
		SourceLedgerVerificationReady:                                            sourceVerification.LedgerVerificationReady,
		SourceLedgerVerificationReason:                                           sourceVerification.Reason,
		SourceLedgerPersistenceSchema:                                            sourceVerification.SourceSchema,
	}
	readiness.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema
	readiness.Status = "shadow_graft_admission_readiness_blocked_dry_run"
	readiness.TargetKind = "weighted_internal_world_shadow_graft_admission_readiness"
	readiness.TargetMode = "closed_readiness_guard_dry_run"
	readiness.Action = "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run"
	readiness.WriterAction = "reject_blocked_ledger_verification"
	readiness.RollbackAction = "reject_blocked_ledger_verification"
	readiness.LedgerState = "blocked"
	readiness.LedgerAction = "reject_blocked_ledger_verification"
	readiness.LedgerContract = "none"
	readiness.LedgerEntrypoint = "none"
	readiness.LedgerReceiptShape = "none"
	readiness.LedgerWriteScope = "none"
	readiness.LedgerReady = false
	readiness.LedgerAppendAllowed = false
	readiness.ReceiptShape = "weighted_resonance_shadow_graft_admission_readiness_receipt"
	readiness.SourceSchema = sourceVerification.Schema
	readiness.SourceStatus = sourceVerification.Status
	readiness.SourceTarget = sourceVerification.Target
	readiness.SourceReport = verificationPath
	readiness.Reason = "weighted resonance shadow graft admission readiness blocked by blocked ledger verification; live admission readiness remains closed"
	readiness.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessCausalID(readiness)
	readiness.AdmissionReadinessHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessHash(readiness)
	readiness.AdmissionReadinessReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReadBackHash(readiness)
	readiness.WeightedAdmissionResonanceGraftAdmissionReadinessID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessID(readiness)
	if readiness.CausalID == "" ||
		readiness.AdmissionReadinessHash == "" ||
		readiness.AdmissionReadinessReadBackHash == "" ||
		readiness.WeightedAdmissionResonanceGraftAdmissionReadinessID == "" ||
		readiness.AdmissionReadinessHash == readiness.AdmissionReadinessReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission readiness read-back proof failed")
	}
	raw, err := json.MarshalIndent(readiness, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission readiness marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission readiness write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-readiness] pass: resonance_graft_admission_readiness_report=%s resonance_graft_admission_ledger_verification_report=%s\n", outputPath, verificationPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-readiness-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission readiness schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema {
		return fmt.Errorf("weighted admission resonance graft admission readiness schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessSchema)
	}
	if report.Status != "shadow_graft_admission_readiness_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission readiness status mismatch: got %q want %q", report.Status, "shadow_graft_admission_readiness_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_readiness" {
		return fmt.Errorf("weighted admission resonance graft admission readiness target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_readiness")
	}
	if report.TargetMode != "closed_readiness_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission readiness target_mode mismatch: got %q want %q", report.TargetMode, "closed_readiness_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission readiness action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_ledger_verification_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_ledger_verification" || report.RollbackAction != "reject_blocked_ledger_verification" {
		return fmt.Errorf("weighted admission resonance graft admission readiness writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_ledger_verification" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission readiness ledger guard mismatch")
	}
	if report.AdmissionReadinessState != "blocked" ||
		report.AdmissionReadinessAction != "reject_blocked_ledger_verification" ||
		report.AdmissionReadinessTarget != "live_admission" ||
		report.AdmissionReadinessTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_verification" ||
		report.AdmissionReadinessTargetMode != "closed_readiness_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission readiness shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_readiness_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission readiness receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_readiness_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_readiness_ready", report.WeightedAdmissionResonanceGraftAdmissionReadinessReady},
		{"weighted_admission_resonance_graft_admission_ledger_verification_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_verification_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired},
		{"next_step_blocked_without_resonance_graft_admission_readiness", report.NextStepBlockedWithoutResonanceGraftAdmissionReadiness},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady},
		{"weighted_admission_resonance_graft_admission_ledger_verification_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady},
		{"weighted_admission_resonance_graft_admission_ledger_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission readiness %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"ledger_verification_append_only", report.LedgerVerificationAppendOnly},
		{"ledger_verification_receipt_read_back", report.LedgerVerificationReceiptReadBack},
		{"ledger_verification_receipt_verified", report.LedgerVerificationReceiptVerified},
		{"ledger_verification_ready", report.LedgerVerificationReady},
		{"admission_readiness_ledger_verified", report.AdmissionReadinessLedgerVerified},
		{"admission_readiness_writer_ready", report.AdmissionReadinessWriterReady},
		{"admission_readiness_rollback_ready", report.AdmissionReadinessRollbackReady},
		{"admission_readiness_ledger_ready", report.AdmissionReadinessLedgerReady},
		{"admission_readiness_ready", report.AdmissionReadinessReady},
		{"source_ledger_verification_append_only", report.SourceLedgerVerificationAppendOnly},
		{"source_ledger_verification_receipt_read_back", report.SourceLedgerVerificationReceiptReadBack},
		{"source_ledger_verification_receipt_verified", report.SourceLedgerVerificationReceiptVerified},
		{"source_ledger_verification_ready", report.SourceLedgerVerificationReady},
		{"ledger_persistence_append_only", report.LedgerPersistenceAppendOnly},
		{"ledger_persistence_receipt_persisted", report.LedgerPersistenceReceiptPersisted},
		{"ledger_persistence_ready", report.LedgerPersistenceReady},
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"writer_contract_present", report.WriterContractPresent},
		{"rollback_contract_present", report.RollbackContractPresent},
		{"ledger_contract_present", report.LedgerContractPresent},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission readiness opened %s", closed.name)
		}
	}
	if !report.AdmissionReadinessDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission readiness admission_readiness_dry_run_only not ready")
	}
	if !report.LedgerVerificationDryRunOnly || !report.SourceLedgerVerificationDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission readiness ledger verification dry-run flag mismatch")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_readiness_id", report.WeightedAdmissionResonanceGraftAdmissionReadinessID},
		{"causal_id", report.CausalID},
		{"admission_readiness_hash", report.AdmissionReadinessHash},
		{"admission_readiness_read_back_hash", report.AdmissionReadinessReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash},
		{"source_weighted_admission_resonance_graft_admission_ledger_verification_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack},
		{"source_ledger_verification_reason", report.SourceLedgerVerificationReason},
		{"source_ledger_persistence_schema", report.SourceLedgerPersistenceSchema},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission readiness %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema {
		return fmt.Errorf("weighted admission resonance graft admission readiness source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_ledger_verification_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission readiness source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_ledger_verification_blocked_dry_run")
	}
	if report.SourceLedgerPersistenceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema {
		return fmt.Errorf("weighted admission resonance graft admission readiness source_ledger_persistence_schema mismatch: got %q want %q", report.SourceLedgerPersistenceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema)
	}
	if report.SourceLedgerVerificationReportReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_verification_receipt" ||
		report.SourceLedgerVerificationState != "blocked" ||
		report.SourceLedgerVerificationAction != "reject_blocked_ledger_persistence" ||
		report.SourceLedgerVerificationTarget != "admission_ledger_receipt" ||
		report.SourceLedgerVerificationTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_persistence" ||
		report.SourceLedgerVerificationTargetMode != "closed_read_back_guard_dry_run" ||
		report.SourceLedgerVerificationReceiptShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission readiness source ledger verification shape mismatch")
	}
	if report.SourceLedgerVerificationReason != "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission readiness source_ledger_verification_reason mismatch: got %q", report.SourceLedgerVerificationReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID, "weighted-resonance-graft-admission-ledger-verification-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID, "weighted-resonance-graft-admission-ledger-verification-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash, "weighted-resonance-graft-admission-ledger-verification-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack, "weighted-resonance-graft-admission-ledger-verification-read-") {
		return fmt.Errorf("weighted admission resonance graft admission readiness source ledger verification mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission readiness body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission readiness causal_id mismatch")
	}
	if report.AdmissionReadinessHash == "" || report.AdmissionReadinessHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission readiness admission_readiness_hash mismatch")
	}
	if report.AdmissionReadinessReadBackHash == "" || report.AdmissionReadinessReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission readiness admission_readiness_read_back_hash mismatch")
	}
	if report.AdmissionReadinessHash == report.AdmissionReadinessReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission readiness read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionReadinessID == "" || report.WeightedAdmissionResonanceGraftAdmissionReadinessID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessID(report) {
		return fmt.Errorf("weighted admission resonance graft admission readiness id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission readiness blocked by blocked ledger verification; live admission readiness remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission readiness reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport) string {
	h := hashJSON(struct {
		SourceVerificationID   string `json:"source_ledger_verification_id"`
		SourceVerificationRead string `json:"source_ledger_verification_read_back_hash"`
		SourcePersistenceID    string `json:"source_ledger_persistence_id"`
		Target                 string `json:"target"`
		State                  string `json:"admission_readiness_state"`
		Action                 string `json:"admission_readiness_action"`
	}{
		SourceVerificationID:   report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		SourceVerificationRead: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack,
		SourcePersistenceID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		Target:                 report.Target,
		State:                  report.AdmissionReadinessState,
		Action:                 report.AdmissionReadinessAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-readiness-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport) string {
	h := hashJSON(struct {
		CausalID                string `json:"causal_id"`
		SourceVerificationID    string `json:"source_ledger_verification_id"`
		SourceVerificationHash  string `json:"source_ledger_verification_hash"`
		SourceVerificationRead  string `json:"source_ledger_verification_read_back_hash"`
		State                   string `json:"admission_readiness_state"`
		Action                  string `json:"admission_readiness_action"`
		Target                  string `json:"admission_readiness_target"`
		TargetKind              string `json:"admission_readiness_target_kind"`
		TargetMode              string `json:"admission_readiness_target_mode"`
		DryRunOnly              bool   `json:"admission_readiness_dry_run_only"`
		LedgerVerified          bool   `json:"admission_readiness_ledger_verified"`
		WriterReady             bool   `json:"admission_readiness_writer_ready"`
		RollbackReady           bool   `json:"admission_readiness_rollback_ready"`
		LedgerReady             bool   `json:"admission_readiness_ledger_ready"`
		Ready                   bool   `json:"admission_readiness_ready"`
		WeightedReady           bool   `json:"weighted_readiness_ready"`
		SourceVerificationReady bool   `json:"source_ledger_verification_ready"`
		SourceReceiptVerified   bool   `json:"source_ledger_verification_receipt_verified"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_readiness"`
	}{
		CausalID:                report.CausalID,
		SourceVerificationID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		SourceVerificationHash:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash,
		SourceVerificationRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack,
		State:                   report.AdmissionReadinessState,
		Action:                  report.AdmissionReadinessAction,
		Target:                  report.AdmissionReadinessTarget,
		TargetKind:              report.AdmissionReadinessTargetKind,
		TargetMode:              report.AdmissionReadinessTargetMode,
		DryRunOnly:              report.AdmissionReadinessDryRunOnly,
		LedgerVerified:          report.AdmissionReadinessLedgerVerified,
		WriterReady:             report.AdmissionReadinessWriterReady,
		RollbackReady:           report.AdmissionReadinessRollbackReady,
		LedgerReady:             report.AdmissionReadinessLedgerReady,
		Ready:                   report.AdmissionReadinessReady,
		WeightedReady:           report.WeightedAdmissionResonanceGraftAdmissionReadinessReady,
		SourceVerificationReady: report.SourceLedgerVerificationReady,
		SourceReceiptVerified:   report.SourceLedgerVerificationReceiptVerified,
		WriteAllowed:            report.WriteAllowed,
		AdmissionAllowed:        report.AdmissionAllowed,
		LiveAdmissionEnabled:    report.LiveAdmissionEnabled,
		MutatesState:            report.MutatesState,
		BodyMutationAllowed:     report.BodyMutationAllowed,
		NextStepBlockedWithout:  report.NextStepBlockedWithoutResonanceGraftAdmissionReadiness,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-readiness-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport) string {
	h := hashJSON(struct {
		AdmissionReadinessHash string `json:"admission_readiness_hash"`
		SourceVerificationID   string `json:"source_ledger_verification_id"`
		SourceVerificationRead string `json:"source_ledger_verification_read_back_hash"`
		WeightedReady          bool   `json:"weighted_readiness_ready"`
		VerificationConsumed   bool   `json:"ledger_verification_consumed"`
		VerificationRequired   bool   `json:"ledger_verification_required"`
		ReadinessReady         bool   `json:"admission_readiness_ready"`
		LedgerVerified         bool   `json:"admission_readiness_ledger_verified"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		AdmissionReadinessHash: report.AdmissionReadinessHash,
		SourceVerificationID:   report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		SourceVerificationRead: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionReadinessReady,
		VerificationConsumed:   report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationConsumed,
		VerificationRequired:   report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationRequired,
		ReadinessReady:         report.AdmissionReadinessReady,
		LedgerVerified:         report.AdmissionReadinessLedgerVerified,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-readiness-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport) string {
	h := hashJSON(struct {
		Schema                     string `json:"schema"`
		Status                     string `json:"status"`
		Action                     string `json:"action"`
		SourceVerificationID       string `json:"source_ledger_verification_id"`
		SourceVerificationHash     string `json:"source_ledger_verification_hash"`
		SourceVerificationRead     string `json:"source_ledger_verification_read_back_hash"`
		CausalID                   string `json:"causal_id"`
		AdmissionReadinessHash     string `json:"admission_readiness_hash"`
		AdmissionReadinessReadBack string `json:"admission_readiness_read_back_hash"`
		State                      string `json:"admission_readiness_state"`
		ActionReadiness            string `json:"admission_readiness_action"`
		Ready                      bool   `json:"weighted_readiness_ready"`
		LedgerVerified             bool   `json:"admission_readiness_ledger_verified"`
		WriteAllowed               bool   `json:"write_allowed"`
		AdmissionAllowed           bool   `json:"admission_allowed"`
		LiveAdmissionEnabled       bool   `json:"live_admission_enabled"`
		MutatesState               bool   `json:"mutates_state"`
		NextStepBlockedWithout     bool   `json:"next_step_blocked_without_resonance_graft_admission_readiness"`
	}{
		Schema:                     report.Schema,
		Status:                     report.Status,
		Action:                     report.Action,
		SourceVerificationID:       report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID,
		SourceVerificationHash:     report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash,
		SourceVerificationRead:     report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBack,
		CausalID:                   report.CausalID,
		AdmissionReadinessHash:     report.AdmissionReadinessHash,
		AdmissionReadinessReadBack: report.AdmissionReadinessReadBackHash,
		State:                      report.AdmissionReadinessState,
		ActionReadiness:            report.AdmissionReadinessAction,
		Ready:                      report.WeightedAdmissionResonanceGraftAdmissionReadinessReady,
		LedgerVerified:             report.AdmissionReadinessLedgerVerified,
		WriteAllowed:               report.WriteAllowed,
		AdmissionAllowed:           report.AdmissionAllowed,
		LiveAdmissionEnabled:       report.LiveAdmissionEnabled,
		MutatesState:               report.MutatesState,
		NextStepBlockedWithout:     report.NextStepBlockedWithoutResonanceGraftAdmissionReadiness,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-readiness-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionReadinessReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission readiness path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission readiness not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission readiness not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission readiness JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission readiness decode failed: %w", err)
	}
	return report, root, nil
}
