package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_persistence.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport

	LedgerPersistenceState            string `json:"ledger_persistence_state"`
	LedgerPersistenceAction           string `json:"ledger_persistence_action"`
	LedgerPersistenceTarget           string `json:"ledger_persistence_target"`
	LedgerPersistenceTargetKind       string `json:"ledger_persistence_target_kind"`
	LedgerPersistenceTargetMode       string `json:"ledger_persistence_target_mode"`
	LedgerPersistenceReceiptShape     string `json:"ledger_persistence_receipt_shape"`
	LedgerPersistenceWriteScope       string `json:"ledger_persistence_write_scope"`
	LedgerPersistenceAppendOnly       bool   `json:"ledger_persistence_append_only"`
	LedgerPersistenceDryRunOnly       bool   `json:"ledger_persistence_dry_run_only"`
	LedgerPersistenceReceiptPersisted bool   `json:"ledger_persistence_receipt_persisted"`
	LedgerPersistenceReady            bool   `json:"ledger_persistence_ready"`

	WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady             bool   `json:"weighted_admission_resonance_graft_admission_ledger_persistence_ready"`
	WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed       bool   `json:"weighted_admission_resonance_graft_admission_ledger_implementation_consumed"`
	WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired       bool   `json:"weighted_admission_resonance_graft_admission_ledger_implementation_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence             bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_persistence"`
	WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID                string `json:"weighted_admission_resonance_graft_admission_ledger_persistence_id"`
	LedgerPersistenceHash                                                      string `json:"ledger_persistence_hash"`
	LedgerPersistenceReadBackHash                                              string `json:"ledger_persistence_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID       string `json:"source_weighted_admission_resonance_graft_admission_ledger_implementation_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady    bool   `json:"source_weighted_admission_resonance_graft_admission_ledger_implementation_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID string `json:"source_weighted_admission_resonance_graft_admission_ledger_implementation_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash     string `json:"source_weighted_admission_resonance_graft_admission_ledger_implementation_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack string `json:"source_weighted_admission_resonance_graft_admission_ledger_implementation_read_back_hash"`
	SourceLedgerImplementationReportReceiptShape                               string `json:"source_ledger_implementation_report_receipt_shape"`
	SourceLedgerImplementationState                                            string `json:"source_ledger_implementation_state"`
	SourceLedgerImplementationAction                                           string `json:"source_ledger_implementation_action"`
	SourceLedgerImplementationTarget                                           string `json:"source_ledger_implementation_target"`
	SourceLedgerImplementationTargetKind                                       string `json:"source_ledger_implementation_target_kind"`
	SourceLedgerImplementationTargetMode                                       string `json:"source_ledger_implementation_target_mode"`
	SourceLedgerImplementationEntrypoint                                       string `json:"source_ledger_implementation_entrypoint"`
	SourceLedgerImplementationReceiptShape                                     string `json:"source_ledger_implementation_receipt_shape"`
	SourceLedgerImplementationWriteScope                                       string `json:"source_ledger_implementation_write_scope"`
	SourceLedgerImplementationAppendOnly                                       bool   `json:"source_ledger_implementation_append_only"`
	SourceLedgerImplementationDryRunOnly                                       bool   `json:"source_ledger_implementation_dry_run_only"`
	SourceLedgerImplementationReceiptPersisted                                 bool   `json:"source_ledger_implementation_receipt_persisted"`
	SourceLedgerImplementationReady                                            bool   `json:"source_ledger_implementation_ready"`
	SourceLedgerImplementationReason                                           string `json:"source_ledger_implementation_reason"`
	SourceAdmissionLedgerSchema                                                string `json:"source_admission_ledger_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistence(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT")
	}
	implementationPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence output path missing")
	}
	sourceImplementation, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportForAssert(implementationPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportError(sourceImplementation, root); err != nil {
		return err
	}
	persistence := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport: sourceImplementation,
		LedgerPersistenceState:                                                     "blocked",
		LedgerPersistenceAction:                                                    "reject_blocked_ledger_implementation",
		LedgerPersistenceTarget:                                                    "admission_ledger_receipt",
		LedgerPersistenceTargetKind:                                                "weighted_internal_world_shadow_graft_admission_ledger_implementation",
		LedgerPersistenceTargetMode:                                                "closed_persistence_guard_dry_run",
		LedgerPersistenceReceiptShape:                                              "none",
		LedgerPersistenceWriteScope:                                                "none",
		LedgerPersistenceAppendOnly:                                                false,
		LedgerPersistenceDryRunOnly:                                                true,
		LedgerPersistenceReceiptPersisted:                                          false,
		LedgerPersistenceReady:                                                     false,
		WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady:             true,
		WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence:             true,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID:       sourceImplementation.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady:    sourceImplementation.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID: sourceImplementation.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash:     sourceImplementation.LedgerImplementationHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack: sourceImplementation.LedgerImplementationReadBackHash,
		SourceLedgerImplementationReportReceiptShape:                               sourceImplementation.ReceiptShape,
		SourceLedgerImplementationState:                                            sourceImplementation.LedgerImplementationState,
		SourceLedgerImplementationAction:                                           sourceImplementation.LedgerImplementationAction,
		SourceLedgerImplementationTarget:                                           sourceImplementation.LedgerImplementationTarget,
		SourceLedgerImplementationTargetKind:                                       sourceImplementation.LedgerImplementationTargetKind,
		SourceLedgerImplementationTargetMode:                                       sourceImplementation.LedgerImplementationTargetMode,
		SourceLedgerImplementationEntrypoint:                                       sourceImplementation.LedgerImplementationEntrypoint,
		SourceLedgerImplementationReceiptShape:                                     sourceImplementation.LedgerImplementationReceiptShape,
		SourceLedgerImplementationWriteScope:                                       sourceImplementation.LedgerImplementationWriteScope,
		SourceLedgerImplementationAppendOnly:                                       sourceImplementation.LedgerImplementationAppendOnly,
		SourceLedgerImplementationDryRunOnly:                                       sourceImplementation.LedgerImplementationDryRunOnly,
		SourceLedgerImplementationReceiptPersisted:                                 sourceImplementation.LedgerImplementationReceiptPersisted,
		SourceLedgerImplementationReady:                                            sourceImplementation.LedgerImplementationReady,
		SourceLedgerImplementationReason:                                           sourceImplementation.Reason,
		SourceAdmissionLedgerSchema:                                                sourceImplementation.SourceSchema,
	}
	persistence.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema
	persistence.Status = "shadow_graft_admission_ledger_persistence_blocked_dry_run"
	persistence.TargetKind = "weighted_internal_world_shadow_graft_admission_ledger_persistence"
	persistence.TargetMode = "closed_ledger_persistence_guard_dry_run"
	persistence.Action = "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run"
	persistence.WriterAction = "reject_blocked_ledger_implementation"
	persistence.RollbackAction = "reject_blocked_ledger_implementation"
	persistence.LedgerState = "blocked"
	persistence.LedgerAction = "reject_blocked_ledger_implementation"
	persistence.LedgerContract = "none"
	persistence.LedgerEntrypoint = "none"
	persistence.LedgerReceiptShape = "none"
	persistence.LedgerWriteScope = "none"
	persistence.LedgerReady = false
	persistence.LedgerAppendAllowed = false
	persistence.ReceiptShape = "weighted_resonance_shadow_graft_admission_ledger_persistence_receipt"
	persistence.SourceSchema = sourceImplementation.Schema
	persistence.SourceStatus = sourceImplementation.Status
	persistence.SourceTarget = sourceImplementation.Target
	persistence.SourceReport = implementationPath
	persistence.Reason = "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed"
	persistence.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID(persistence)
	persistence.LedgerPersistenceHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash(persistence)
	persistence.LedgerPersistenceReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBackHash(persistence)
	persistence.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID(persistence)
	if persistence.CausalID == "" ||
		persistence.LedgerPersistenceHash == "" ||
		persistence.LedgerPersistenceReadBackHash == "" ||
		persistence.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID == "" ||
		persistence.LedgerPersistenceHash == persistence.LedgerPersistenceReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence read-back proof failed")
	}
	raw, err := json.MarshalIndent(persistence, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence] pass: resonance_graft_admission_ledger_persistence_report=%s resonance_graft_admission_ledger_implementation_report=%s\n", outputPath, implementationPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-persistence-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema)
	}
	if report.Status != "shadow_graft_admission_ledger_persistence_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence status mismatch: got %q want %q", report.Status, "shadow_graft_admission_ledger_persistence_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_persistence" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_ledger_persistence")
	}
	if report.TargetMode != "closed_ledger_persistence_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence target_mode mismatch: got %q want %q", report.TargetMode, "closed_ledger_persistence_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_ledger_implementation_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_ledger_implementation" || report.RollbackAction != "reject_blocked_ledger_implementation" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_ledger_implementation" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence ledger guard mismatch")
	}
	if report.LedgerPersistenceState != "blocked" ||
		report.LedgerPersistenceAction != "reject_blocked_ledger_implementation" ||
		report.LedgerPersistenceTarget != "admission_ledger_receipt" ||
		report.LedgerPersistenceTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_implementation" ||
		report.LedgerPersistenceTargetMode != "closed_persistence_guard_dry_run" ||
		report.LedgerPersistenceReceiptShape != "none" ||
		report.LedgerPersistenceWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_persistence_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_ledger_persistence_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_ledger_persistence_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired},
		{"next_step_blocked_without_resonance_graft_admission_ledger_persistence", report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady},
		{"weighted_admission_resonance_graft_admission_ledger_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerRequired},
		{"source_weighted_admission_resonance_graft_admission_ledger_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReady},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger persistence %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"ledger_ready", report.LedgerReady},
		{"ledger_append_allowed", report.LedgerAppendAllowed},
		{"ledger_implementation_append_only", report.LedgerImplementationAppendOnly},
		{"ledger_implementation_receipt_persisted", report.LedgerImplementationReceiptPersisted},
		{"ledger_implementation_ready", report.LedgerImplementationReady},
		{"ledger_persistence_append_only", report.LedgerPersistenceAppendOnly},
		{"ledger_persistence_receipt_persisted", report.LedgerPersistenceReceiptPersisted},
		{"ledger_persistence_ready", report.LedgerPersistenceReady},
		{"source_ledger_implementation_append_only", report.SourceLedgerImplementationAppendOnly},
		{"source_ledger_implementation_receipt_persisted", report.SourceLedgerImplementationReceiptPersisted},
		{"source_ledger_implementation_ready", report.SourceLedgerImplementationReady},
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"writer_contract_present", report.WriterContractPresent},
		{"rollback_contract_present", report.RollbackContractPresent},
		{"ledger_contract_present", report.LedgerContractPresent},
		{"source_admission_ledger_ledger_ready", report.SourceAdmissionLedgerLedgerReady},
		{"source_admission_ledger_ledger_append_allowed", report.SourceAdmissionLedgerLedgerAppendAllowed},
		{"source_writer_contract_contracts_ready", report.SourceWriterContractContractsReady},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"authority_granted", report.AuthorityGranted},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission ledger persistence opened %s", closed.name)
		}
	}
	if !report.LedgerPersistenceDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence ledger_persistence_dry_run_only not ready")
	}
	if !report.SourceLedgerImplementationDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source_ledger_implementation_dry_run_only not ready")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_ledger_persistence_id", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID},
		{"causal_id", report.CausalID},
		{"ledger_persistence_hash", report.LedgerPersistenceHash},
		{"ledger_persistence_read_back_hash", report.LedgerPersistenceReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack},
		{"source_ledger_implementation_reason", report.SourceLedgerImplementationReason},
		{"source_admission_ledger_schema", report.SourceAdmissionLedgerSchema},
		{"source_weighted_admission_resonance_graft_admission_ledger_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission ledger persistence %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_ledger_implementation_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_ledger_implementation_blocked_dry_run")
	}
	if report.SourceAdmissionLedgerSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source_admission_ledger_schema mismatch: got %q want %q", report.SourceAdmissionLedgerSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema)
	}
	if report.SourceLedgerImplementationReportReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_implementation_receipt" ||
		report.SourceLedgerImplementationState != "blocked" ||
		report.SourceLedgerImplementationAction != "reject_blocked_admission_ledger" ||
		report.SourceLedgerImplementationTarget != "admission_ledger" ||
		report.SourceLedgerImplementationTargetKind != "weighted_internal_world_shadow_graft_admission_ledger" ||
		report.SourceLedgerImplementationTargetMode != "closed_append_guard_dry_run" ||
		report.SourceLedgerImplementationEntrypoint != "none" ||
		report.SourceLedgerImplementationReceiptShape != "none" ||
		report.SourceLedgerImplementationWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source ledger implementation shape mismatch")
	}
	if report.SourceLedgerImplementationReason != "weighted resonance shadow graft admission ledger implementation blocked by blocked admission ledger; implementation append contract remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source_ledger_implementation_reason mismatch: got %q", report.SourceLedgerImplementationReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID, "weighted-resonance-graft-admission-ledger-implementation-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID, "weighted-resonance-graft-admission-ledger-implementation-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash, "weighted-resonance-graft-admission-ledger-implementation-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack, "weighted-resonance-graft-admission-ledger-implementation-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence source ledger implementation mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence causal_id mismatch")
	}
	if report.LedgerPersistenceHash == "" || report.LedgerPersistenceHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence ledger_persistence_hash mismatch")
	}
	if report.LedgerPersistenceReadBackHash == "" || report.LedgerPersistenceReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence ledger_persistence_read_back_hash mismatch")
	}
	if report.LedgerPersistenceHash == report.LedgerPersistenceReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID == "" || report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger persistence reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport) string {
	h := hashJSON(struct {
		SourceImplementationID   string `json:"source_ledger_implementation_id"`
		SourceImplementationRead string `json:"source_ledger_implementation_read_back_hash"`
		SourceLedgerID           string `json:"source_ledger_id"`
		Target                   string `json:"target"`
		State                    string `json:"ledger_persistence_state"`
		Action                   string `json:"ledger_persistence_action"`
	}{
		SourceImplementationID:   report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		SourceImplementationRead: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack,
		SourceLedgerID:           report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		Target:                   report.Target,
		State:                    report.LedgerPersistenceState,
		Action:                   report.LedgerPersistenceAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-persistence-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport) string {
	h := hashJSON(struct {
		CausalID                  string `json:"causal_id"`
		SourceImplementationID    string `json:"source_ledger_implementation_id"`
		SourceImplementationHash  string `json:"source_ledger_implementation_hash"`
		SourceImplementationRead  string `json:"source_ledger_implementation_read_back_hash"`
		SourceLedgerID            string `json:"source_ledger_id"`
		State                     string `json:"ledger_persistence_state"`
		Action                    string `json:"ledger_persistence_action"`
		Target                    string `json:"ledger_persistence_target"`
		TargetKind                string `json:"ledger_persistence_target_kind"`
		TargetMode                string `json:"ledger_persistence_target_mode"`
		ReceiptShape              string `json:"ledger_persistence_receipt_shape"`
		WriteScope                string `json:"ledger_persistence_write_scope"`
		AppendOnly                bool   `json:"ledger_persistence_append_only"`
		DryRunOnly                bool   `json:"ledger_persistence_dry_run_only"`
		ReceiptPersisted          bool   `json:"ledger_persistence_receipt_persisted"`
		Ready                     bool   `json:"ledger_persistence_ready"`
		WeightedReady             bool   `json:"weighted_ledger_persistence_ready"`
		SourceImplementationReady bool   `json:"source_ledger_implementation_ready"`
		LedgerAppendAllowed       bool   `json:"ledger_append_allowed"`
		WriteAllowed              bool   `json:"write_allowed"`
		AdmissionAllowed          bool   `json:"admission_allowed"`
		LiveAdmissionEnabled      bool   `json:"live_admission_enabled"`
		MutatesState              bool   `json:"mutates_state"`
		BodyMutationAllowed       bool   `json:"body_mutation_allowed"`
		ContractsReady            bool   `json:"contracts_ready"`
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_persistence"`
	}{
		CausalID:                  report.CausalID,
		SourceImplementationID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		SourceImplementationHash:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash,
		SourceImplementationRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack,
		SourceLedgerID:            report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		State:                     report.LedgerPersistenceState,
		Action:                    report.LedgerPersistenceAction,
		Target:                    report.LedgerPersistenceTarget,
		TargetKind:                report.LedgerPersistenceTargetKind,
		TargetMode:                report.LedgerPersistenceTargetMode,
		ReceiptShape:              report.LedgerPersistenceReceiptShape,
		WriteScope:                report.LedgerPersistenceWriteScope,
		AppendOnly:                report.LedgerPersistenceAppendOnly,
		DryRunOnly:                report.LedgerPersistenceDryRunOnly,
		ReceiptPersisted:          report.LedgerPersistenceReceiptPersisted,
		Ready:                     report.LedgerPersistenceReady,
		WeightedReady:             report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady,
		SourceImplementationReady: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady,
		LedgerAppendAllowed:       report.LedgerAppendAllowed,
		WriteAllowed:              report.WriteAllowed,
		AdmissionAllowed:          report.AdmissionAllowed,
		LiveAdmissionEnabled:      report.LiveAdmissionEnabled,
		MutatesState:              report.MutatesState,
		BodyMutationAllowed:       report.BodyMutationAllowed,
		ContractsReady:            report.ContractsReady,
		NextStepBlockedWithout:    report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-persistence-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport) string {
	h := hashJSON(struct {
		LedgerPersistenceHash    string `json:"ledger_persistence_hash"`
		SourceImplementationID   string `json:"source_ledger_implementation_id"`
		SourceImplementationRead string `json:"source_ledger_implementation_read_back_hash"`
		WeightedReady            bool   `json:"weighted_ledger_persistence_ready"`
		ImplementationConsumed   bool   `json:"ledger_implementation_consumed"`
		ImplementationRequired   bool   `json:"ledger_implementation_required"`
		PersistenceReady         bool   `json:"ledger_persistence_ready"`
		ReceiptPersisted         bool   `json:"ledger_persistence_receipt_persisted"`
		ContractsReady           bool   `json:"contracts_ready"`
		WriteAllowed             bool   `json:"write_allowed"`
		AdmissionAllowed         bool   `json:"admission_allowed"`
		LiveAdmissionEnabled     bool   `json:"live_admission_enabled"`
		MutatesState             bool   `json:"mutates_state"`
	}{
		LedgerPersistenceHash:    report.LedgerPersistenceHash,
		SourceImplementationID:   report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		SourceImplementationRead: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack,
		WeightedReady:            report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady,
		ImplementationConsumed:   report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed,
		ImplementationRequired:   report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired,
		PersistenceReady:         report.LedgerPersistenceReady,
		ReceiptPersisted:         report.LedgerPersistenceReceiptPersisted,
		ContractsReady:           report.ContractsReady,
		WriteAllowed:             report.WriteAllowed,
		AdmissionAllowed:         report.AdmissionAllowed,
		LiveAdmissionEnabled:     report.LiveAdmissionEnabled,
		MutatesState:             report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-persistence-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport) string {
	h := hashJSON(struct {
		Schema                    string `json:"schema"`
		Status                    string `json:"status"`
		Action                    string `json:"action"`
		SourceImplementationID    string `json:"source_ledger_implementation_id"`
		SourceImplementationHash  string `json:"source_ledger_implementation_hash"`
		SourceImplementationRead  string `json:"source_ledger_implementation_read_back_hash"`
		SourceLedgerID            string `json:"source_ledger_id"`
		CausalID                  string `json:"causal_id"`
		LedgerPersistenceHash     string `json:"ledger_persistence_hash"`
		LedgerPersistenceReadBack string `json:"ledger_persistence_read_back_hash"`
		State                     string `json:"ledger_persistence_state"`
		ActionPersistence         string `json:"ledger_persistence_action"`
		Ready                     bool   `json:"weighted_ledger_persistence_ready"`
		ReceiptPersisted          bool   `json:"ledger_persistence_receipt_persisted"`
		WriteAllowed              bool   `json:"write_allowed"`
		AdmissionAllowed          bool   `json:"admission_allowed"`
		LiveAdmissionEnabled      bool   `json:"live_admission_enabled"`
		MutatesState              bool   `json:"mutates_state"`
		NextStepBlockedWithout    bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_persistence"`
	}{
		Schema:                    report.Schema,
		Status:                    report.Status,
		Action:                    report.Action,
		SourceImplementationID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		SourceImplementationHash:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash,
		SourceImplementationRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBack,
		SourceLedgerID:            report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		CausalID:                  report.CausalID,
		LedgerPersistenceHash:     report.LedgerPersistenceHash,
		LedgerPersistenceReadBack: report.LedgerPersistenceReadBackHash,
		State:                     report.LedgerPersistenceState,
		ActionPersistence:         report.LedgerPersistenceAction,
		Ready:                     report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady,
		ReceiptPersisted:          report.LedgerPersistenceReceiptPersisted,
		WriteAllowed:              report.WriteAllowed,
		AdmissionAllowed:          report.AdmissionAllowed,
		LiveAdmissionEnabled:      report.LiveAdmissionEnabled,
		MutatesState:              report.MutatesState,
		NextStepBlockedWithout:    report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerPersistence,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-persistence-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger persistence path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger persistence not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger persistence not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger persistence JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger persistence decode failed: %w", err)
	}
	return report, root, nil
}
