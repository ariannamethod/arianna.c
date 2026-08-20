package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_implementation.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport

	LedgerImplementationState            string `json:"ledger_implementation_state"`
	LedgerImplementationAction           string `json:"ledger_implementation_action"`
	LedgerImplementationTarget           string `json:"ledger_implementation_target"`
	LedgerImplementationTargetKind       string `json:"ledger_implementation_target_kind"`
	LedgerImplementationTargetMode       string `json:"ledger_implementation_target_mode"`
	LedgerImplementationEntrypoint       string `json:"ledger_implementation_entrypoint"`
	LedgerImplementationReceiptShape     string `json:"ledger_implementation_receipt_shape"`
	LedgerImplementationWriteScope       string `json:"ledger_implementation_write_scope"`
	LedgerImplementationAppendOnly       bool   `json:"ledger_implementation_append_only"`
	LedgerImplementationDryRunOnly       bool   `json:"ledger_implementation_dry_run_only"`
	LedgerImplementationReceiptPersisted bool   `json:"ledger_implementation_receipt_persisted"`
	LedgerImplementationReady            bool   `json:"ledger_implementation_ready"`

	WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady bool   `json:"weighted_admission_resonance_graft_admission_ledger_implementation_ready"`
	WeightedAdmissionResonanceGraftAdmissionLedgerConsumed            bool   `json:"weighted_admission_resonance_graft_admission_ledger_consumed"`
	WeightedAdmissionResonanceGraftAdmissionLedgerRequired            bool   `json:"weighted_admission_resonance_graft_admission_ledger_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_implementation"`
	WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID    string `json:"weighted_admission_resonance_graft_admission_ledger_implementation_id"`
	LedgerImplementationHash                                          string `json:"ledger_implementation_hash"`
	LedgerImplementationReadBackHash                                  string `json:"ledger_implementation_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerID            string `json:"source_weighted_admission_resonance_graft_admission_ledger_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerReady         bool   `json:"source_weighted_admission_resonance_graft_admission_ledger_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerCausalID      string `json:"source_weighted_admission_resonance_graft_admission_ledger_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash          string `json:"source_weighted_admission_resonance_graft_admission_ledger_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack      string `json:"source_weighted_admission_resonance_graft_admission_ledger_read_back_hash"`
	SourceAdmissionLedgerReceiptShape                                 string `json:"source_admission_ledger_receipt_shape"`
	SourceAdmissionLedgerKind                                         string `json:"source_admission_ledger_kind"`
	SourceAdmissionLedgerMode                                         string `json:"source_admission_ledger_mode"`
	SourceAdmissionLedgerStage                                        string `json:"source_admission_ledger_stage"`
	SourceAdmissionLedgerLedgerState                                  string `json:"source_admission_ledger_ledger_state"`
	SourceAdmissionLedgerLedgerAction                                 string `json:"source_admission_ledger_ledger_action"`
	SourceAdmissionLedgerLedgerContract                               string `json:"source_admission_ledger_ledger_contract"`
	SourceAdmissionLedgerLedgerEntrypoint                             string `json:"source_admission_ledger_ledger_entrypoint"`
	SourceAdmissionLedgerLedgerReceiptShape                           string `json:"source_admission_ledger_ledger_receipt_shape"`
	SourceAdmissionLedgerLedgerWriteScope                             string `json:"source_admission_ledger_ledger_write_scope"`
	SourceAdmissionLedgerLedgerReady                                  bool   `json:"source_admission_ledger_ledger_ready"`
	SourceAdmissionLedgerLedgerAppendAllowed                          bool   `json:"source_admission_ledger_ledger_append_allowed"`
	SourceAdmissionLedgerReason                                       string `json:"source_admission_ledger_reason"`
	SourceWriterContractSchema                                        string `json:"source_writer_contract_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementation(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation RESONANCE_GRAFT_ADMISSION_LEDGER_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_IMPLEMENTATION_REPORT")
	}
	ledgerPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation output path missing")
	}
	sourceLedger, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportForAssert(ledgerPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReportError(sourceLedger, root); err != nil {
		return err
	}
	impl := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerReport: sourceLedger,
		LedgerImplementationState:                                         "blocked",
		LedgerImplementationAction:                                        "reject_blocked_admission_ledger",
		LedgerImplementationTarget:                                        "admission_ledger",
		LedgerImplementationTargetKind:                                    "weighted_internal_world_shadow_graft_admission_ledger",
		LedgerImplementationTargetMode:                                    "closed_append_guard_dry_run",
		LedgerImplementationEntrypoint:                                    "none",
		LedgerImplementationReceiptShape:                                  "none",
		LedgerImplementationWriteScope:                                    "none",
		LedgerImplementationAppendOnly:                                    false,
		LedgerImplementationDryRunOnly:                                    true,
		LedgerImplementationReceiptPersisted:                              false,
		LedgerImplementationReady:                                         false,
		WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady: true,
		WeightedAdmissionResonanceGraftAdmissionLedgerConsumed:            true,
		WeightedAdmissionResonanceGraftAdmissionLedgerRequired:            true,
		NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation: true,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerID:            sourceLedger.WeightedAdmissionResonanceGraftAdmissionLedgerID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerReady:         sourceLedger.WeightedAdmissionResonanceGraftAdmissionLedgerReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerCausalID:      sourceLedger.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash:          sourceLedger.AdmissionLedgerHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack:      sourceLedger.ReadBackHash,
		SourceAdmissionLedgerReceiptShape:                                 sourceLedger.ReceiptShape,
		SourceAdmissionLedgerKind:                                         sourceLedger.AdmissionLedgerKind,
		SourceAdmissionLedgerMode:                                         sourceLedger.AdmissionLedgerMode,
		SourceAdmissionLedgerStage:                                        sourceLedger.AdmissionLedgerStage,
		SourceAdmissionLedgerLedgerState:                                  sourceLedger.LedgerState,
		SourceAdmissionLedgerLedgerAction:                                 sourceLedger.LedgerAction,
		SourceAdmissionLedgerLedgerContract:                               sourceLedger.LedgerContract,
		SourceAdmissionLedgerLedgerEntrypoint:                             sourceLedger.LedgerEntrypoint,
		SourceAdmissionLedgerLedgerReceiptShape:                           sourceLedger.LedgerReceiptShape,
		SourceAdmissionLedgerLedgerWriteScope:                             sourceLedger.LedgerWriteScope,
		SourceAdmissionLedgerLedgerReady:                                  sourceLedger.LedgerReady,
		SourceAdmissionLedgerLedgerAppendAllowed:                          sourceLedger.LedgerAppendAllowed,
		SourceAdmissionLedgerReason:                                       sourceLedger.Reason,
		SourceWriterContractSchema:                                        sourceLedger.SourceSchema,
	}
	impl.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema
	impl.Status = "shadow_graft_admission_ledger_implementation_blocked_dry_run"
	impl.TargetKind = "weighted_internal_world_shadow_graft_admission_ledger_implementation"
	impl.TargetMode = "closed_ledger_implementation_guard_dry_run"
	impl.Action = "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run"
	impl.WriterAction = "reject_blocked_admission_ledger"
	impl.RollbackAction = "reject_blocked_admission_ledger"
	impl.LedgerState = "blocked"
	impl.LedgerAction = "reject_blocked_admission_ledger"
	impl.LedgerContract = "none"
	impl.LedgerEntrypoint = "none"
	impl.LedgerReceiptShape = "none"
	impl.LedgerWriteScope = "none"
	impl.LedgerReady = false
	impl.LedgerAppendAllowed = false
	impl.ReceiptShape = "weighted_resonance_shadow_graft_admission_ledger_implementation_receipt"
	impl.SourceSchema = sourceLedger.Schema
	impl.SourceStatus = sourceLedger.Status
	impl.SourceTarget = sourceLedger.Target
	impl.SourceReport = ledgerPath
	impl.Reason = "weighted resonance shadow graft admission ledger implementation blocked by blocked admission ledger; implementation append contract remains closed"
	impl.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID(impl)
	impl.LedgerImplementationHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash(impl)
	impl.LedgerImplementationReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBackHash(impl)
	impl.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID(impl)
	if impl.CausalID == "" ||
		impl.LedgerImplementationHash == "" ||
		impl.LedgerImplementationReadBackHash == "" ||
		impl.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID == "" ||
		impl.LedgerImplementationHash == impl.LedgerImplementationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation read-back proof failed")
	}
	raw, err := json.MarshalIndent(impl, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation] pass: resonance_graft_admission_ledger_implementation_report=%s resonance_graft_admission_ledger_report=%s\n", outputPath, ledgerPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-implementation-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema)
	}
	if report.Status != "shadow_graft_admission_ledger_implementation_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation status mismatch: got %q want %q", report.Status, "shadow_graft_admission_ledger_implementation_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_implementation" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_ledger_implementation")
	}
	if report.TargetMode != "closed_ledger_implementation_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation target_mode mismatch: got %q want %q", report.TargetMode, "closed_ledger_implementation_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_ledger_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_admission_ledger" || report.RollbackAction != "reject_blocked_admission_ledger" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_admission_ledger" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation ledger guard mismatch")
	}
	if report.LedgerImplementationState != "blocked" ||
		report.LedgerImplementationAction != "reject_blocked_admission_ledger" ||
		report.LedgerImplementationTarget != "admission_ledger" ||
		report.LedgerImplementationTargetKind != "weighted_internal_world_shadow_graft_admission_ledger" ||
		report.LedgerImplementationTargetMode != "closed_append_guard_dry_run" ||
		report.LedgerImplementationEntrypoint != "none" ||
		report.LedgerImplementationReceiptShape != "none" ||
		report.LedgerImplementationWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_implementation_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_ledger_implementation_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_ledger_implementation_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady},
		{"weighted_admission_resonance_graft_admission_ledger_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerRequired},
		{"next_step_blocked_without_resonance_graft_admission_ledger_implementation", report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation},
		{"source_weighted_admission_resonance_graft_admission_ledger_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReady},
		{"weighted_admission_resonance_graft_admission_writer_contract_consumed", report.WeightedAdmissionResonanceGraftAdmissionWriterContractConsumed},
		{"weighted_admission_resonance_graft_admission_writer_contract_required", report.WeightedAdmissionResonanceGraftAdmissionWriterContractRequired},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractReady},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger implementation %s not ready", required.name)
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
			return fmt.Errorf("weighted admission resonance graft admission ledger implementation opened %s", closed.name)
		}
	}
	if !report.LedgerImplementationDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation ledger_implementation_dry_run_only not ready")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_ledger_implementation_id", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID},
		{"causal_id", report.CausalID},
		{"ledger_implementation_hash", report.LedgerImplementationHash},
		{"ledger_implementation_read_back_hash", report.LedgerImplementationReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_ledger_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID},
		{"source_weighted_admission_resonance_graft_admission_ledger_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerCausalID},
		{"source_weighted_admission_resonance_graft_admission_ledger_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash},
		{"source_weighted_admission_resonance_graft_admission_ledger_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack},
		{"source_admission_ledger_reason", report.SourceAdmissionLedgerReason},
		{"source_writer_contract_schema", report.SourceWriterContractSchema},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID},
		{"source_weighted_admission_resonance_graft_admission_writer_contract_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash},
		{"source_weighted_admission_resonance_graft_admission_writer_inventory_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterInventoryID},
		{"source_weighted_admission_resonance_graft_admission_writer_preflight_id", report.SourceWeightedAdmissionResonanceGraftAdmissionWriterPreflightID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission ledger implementation %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_ledger_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_ledger_blocked_dry_run")
	}
	if report.SourceAdmissionLedgerReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_receipt" ||
		report.SourceAdmissionLedgerKind != "shadow_graft_admission_ledger" ||
		report.SourceAdmissionLedgerMode != "closed_writer_contract_ledger_guard" ||
		report.SourceAdmissionLedgerStage != "pre_ledger_append_graft_admission_ledger" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source admission ledger shape mismatch")
	}
	if report.SourceAdmissionLedgerLedgerState != "blocked" ||
		report.SourceAdmissionLedgerLedgerAction != "reject_blocked_writer_contract" ||
		report.SourceAdmissionLedgerLedgerContract != "none" ||
		report.SourceAdmissionLedgerLedgerEntrypoint != "none" ||
		report.SourceAdmissionLedgerLedgerReceiptShape != "none" ||
		report.SourceAdmissionLedgerLedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source admission ledger guard mismatch")
	}
	if report.SourceAdmissionLedgerReason != "weighted resonance shadow graft admission ledger blocked by blocked writer contract; ledger receipt append remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source_admission_ledger_reason mismatch: got %q", report.SourceAdmissionLedgerReason)
	}
	if report.SourceWriterContractSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source_writer_contract_schema mismatch: got %q want %q", report.SourceWriterContractSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterContractSchema)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID, "weighted-resonance-graft-admission-ledger-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerCausalID, "weighted-resonance-graft-admission-ledger-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash, "weighted-resonance-graft-admission-ledger-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack, "weighted-resonance-graft-admission-ledger-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source admission ledger mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID, "weighted-resonance-graft-admission-writer-contract-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash, "weighted-resonance-graft-admission-writer-contract-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation source writer contract mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation causal_id mismatch")
	}
	if report.LedgerImplementationHash == "" || report.LedgerImplementationHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation ledger_implementation_hash mismatch")
	}
	if report.LedgerImplementationReadBackHash == "" || report.LedgerImplementationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation ledger_implementation_read_back_hash mismatch")
	}
	if report.LedgerImplementationHash == report.LedgerImplementationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID == "" || report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission ledger implementation blocked by blocked admission ledger; implementation append contract remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger implementation reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport) string {
	h := hashJSON(struct {
		SourceLedgerID       string `json:"source_ledger_id"`
		SourceLedgerRead     string `json:"source_ledger_read_back_hash"`
		SourceWriterContract string `json:"source_writer_contract_id"`
		Target               string `json:"target"`
		State                string `json:"ledger_implementation_state"`
		Action               string `json:"ledger_implementation_action"`
	}{
		SourceLedgerID:       report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		SourceLedgerRead:     report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack,
		SourceWriterContract: report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		Target:               report.Target,
		State:                report.LedgerImplementationState,
		Action:               report.LedgerImplementationAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-implementation-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport) string {
	h := hashJSON(struct {
		CausalID                 string `json:"causal_id"`
		SourceLedgerID           string `json:"source_ledger_id"`
		SourceLedgerHash         string `json:"source_ledger_hash"`
		SourceLedgerRead         string `json:"source_ledger_read_back_hash"`
		SourceWriterContractID   string `json:"source_writer_contract_id"`
		SourceWriterContractHash string `json:"source_writer_contract_hash"`
		State                    string `json:"ledger_implementation_state"`
		Action                   string `json:"ledger_implementation_action"`
		Target                   string `json:"ledger_implementation_target"`
		TargetKind               string `json:"ledger_implementation_target_kind"`
		TargetMode               string `json:"ledger_implementation_target_mode"`
		Entrypoint               string `json:"ledger_implementation_entrypoint"`
		ReceiptShape             string `json:"ledger_implementation_receipt_shape"`
		WriteScope               string `json:"ledger_implementation_write_scope"`
		AppendOnly               bool   `json:"ledger_implementation_append_only"`
		DryRunOnly               bool   `json:"ledger_implementation_dry_run_only"`
		ReceiptPersisted         bool   `json:"ledger_implementation_receipt_persisted"`
		Ready                    bool   `json:"ledger_implementation_ready"`
		WeightedReady            bool   `json:"weighted_ledger_implementation_ready"`
		LedgerAppendAllowed      bool   `json:"ledger_append_allowed"`
		WriteAllowed             bool   `json:"write_allowed"`
		AdmissionAllowed         bool   `json:"admission_allowed"`
		LiveAdmissionEnabled     bool   `json:"live_admission_enabled"`
		MutatesState             bool   `json:"mutates_state"`
		BodyMutationAllowed      bool   `json:"body_mutation_allowed"`
		ContractsReady           bool   `json:"contracts_ready"`
		NextStepBlockedWithout   bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_implementation"`
	}{
		CausalID:                 report.CausalID,
		SourceLedgerID:           report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		SourceLedgerHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash,
		SourceLedgerRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack,
		SourceWriterContractID:   report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		SourceWriterContractHash: report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractHash,
		State:                    report.LedgerImplementationState,
		Action:                   report.LedgerImplementationAction,
		Target:                   report.LedgerImplementationTarget,
		TargetKind:               report.LedgerImplementationTargetKind,
		TargetMode:               report.LedgerImplementationTargetMode,
		Entrypoint:               report.LedgerImplementationEntrypoint,
		ReceiptShape:             report.LedgerImplementationReceiptShape,
		WriteScope:               report.LedgerImplementationWriteScope,
		AppendOnly:               report.LedgerImplementationAppendOnly,
		DryRunOnly:               report.LedgerImplementationDryRunOnly,
		ReceiptPersisted:         report.LedgerImplementationReceiptPersisted,
		Ready:                    report.LedgerImplementationReady,
		WeightedReady:            report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady,
		LedgerAppendAllowed:      report.LedgerAppendAllowed,
		WriteAllowed:             report.WriteAllowed,
		AdmissionAllowed:         report.AdmissionAllowed,
		LiveAdmissionEnabled:     report.LiveAdmissionEnabled,
		MutatesState:             report.MutatesState,
		BodyMutationAllowed:      report.BodyMutationAllowed,
		ContractsReady:           report.ContractsReady,
		NextStepBlockedWithout:   report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-implementation-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport) string {
	h := hashJSON(struct {
		LedgerImplementationHash string `json:"ledger_implementation_hash"`
		SourceLedgerID           string `json:"source_ledger_id"`
		SourceLedgerRead         string `json:"source_ledger_read_back_hash"`
		WeightedReady            bool   `json:"weighted_ledger_implementation_ready"`
		LedgerConsumed           bool   `json:"ledger_consumed"`
		LedgerRequired           bool   `json:"ledger_required"`
		ImplementationReady      bool   `json:"ledger_implementation_ready"`
		LedgerAppendAllowed      bool   `json:"ledger_append_allowed"`
		ContractsReady           bool   `json:"contracts_ready"`
		WriteAllowed             bool   `json:"write_allowed"`
		AdmissionAllowed         bool   `json:"admission_allowed"`
		LiveAdmissionEnabled     bool   `json:"live_admission_enabled"`
		MutatesState             bool   `json:"mutates_state"`
	}{
		LedgerImplementationHash: report.LedgerImplementationHash,
		SourceLedgerID:           report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		SourceLedgerRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack,
		WeightedReady:            report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady,
		LedgerConsumed:           report.WeightedAdmissionResonanceGraftAdmissionLedgerConsumed,
		LedgerRequired:           report.WeightedAdmissionResonanceGraftAdmissionLedgerRequired,
		ImplementationReady:      report.LedgerImplementationReady,
		LedgerAppendAllowed:      report.LedgerAppendAllowed,
		ContractsReady:           report.ContractsReady,
		WriteAllowed:             report.WriteAllowed,
		AdmissionAllowed:         report.AdmissionAllowed,
		LiveAdmissionEnabled:     report.LiveAdmissionEnabled,
		MutatesState:             report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-implementation-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport) string {
	h := hashJSON(struct {
		Schema                   string `json:"schema"`
		Status                   string `json:"status"`
		Action                   string `json:"action"`
		SourceLedgerID           string `json:"source_ledger_id"`
		SourceLedgerHash         string `json:"source_ledger_hash"`
		SourceLedgerRead         string `json:"source_ledger_read_back_hash"`
		SourceWriterContractID   string `json:"source_writer_contract_id"`
		CausalID                 string `json:"causal_id"`
		LedgerImplementationHash string `json:"ledger_implementation_hash"`
		LedgerReadBackHash       string `json:"ledger_implementation_read_back_hash"`
		State                    string `json:"ledger_implementation_state"`
		ActionImpl               string `json:"ledger_implementation_action"`
		Entrypoint               string `json:"ledger_implementation_entrypoint"`
		Ready                    bool   `json:"weighted_ledger_implementation_ready"`
		AppendAllowed            bool   `json:"ledger_append_allowed"`
		WriteAllowed             bool   `json:"write_allowed"`
		AdmissionAllowed         bool   `json:"admission_allowed"`
		LiveAdmissionEnabled     bool   `json:"live_admission_enabled"`
		MutatesState             bool   `json:"mutates_state"`
		NextStepBlockedWithout   bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_implementation"`
	}{
		Schema:                   report.Schema,
		Status:                   report.Status,
		Action:                   report.Action,
		SourceLedgerID:           report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID,
		SourceLedgerHash:         report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerHash,
		SourceLedgerRead:         report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerReadBack,
		SourceWriterContractID:   report.SourceWeightedAdmissionResonanceGraftAdmissionWriterContractID,
		CausalID:                 report.CausalID,
		LedgerImplementationHash: report.LedgerImplementationHash,
		LedgerReadBackHash:       report.LedgerImplementationReadBackHash,
		State:                    report.LedgerImplementationState,
		ActionImpl:               report.LedgerImplementationAction,
		Entrypoint:               report.LedgerImplementationEntrypoint,
		Ready:                    report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationReady,
		AppendAllowed:            report.LedgerAppendAllowed,
		WriteAllowed:             report.WriteAllowed,
		AdmissionAllowed:         report.AdmissionAllowed,
		LiveAdmissionEnabled:     report.LiveAdmissionEnabled,
		MutatesState:             report.MutatesState,
		NextStepBlockedWithout:   report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerImplementation,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-implementation-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger implementation path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger implementation not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger implementation not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger implementation JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger implementation decode failed: %w", err)
	}
	return report, root, nil
}
