package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_ledger_verification.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport struct {
	admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport

	LedgerVerificationState           string `json:"ledger_verification_state"`
	LedgerVerificationAction          string `json:"ledger_verification_action"`
	LedgerVerificationTarget          string `json:"ledger_verification_target"`
	LedgerVerificationTargetKind      string `json:"ledger_verification_target_kind"`
	LedgerVerificationTargetMode      string `json:"ledger_verification_target_mode"`
	LedgerVerificationReceiptShape    string `json:"ledger_verification_receipt_shape"`
	LedgerVerificationAppendOnly      bool   `json:"ledger_verification_append_only"`
	LedgerVerificationDryRunOnly      bool   `json:"ledger_verification_dry_run_only"`
	LedgerVerificationReceiptReadBack bool   `json:"ledger_verification_receipt_read_back"`
	LedgerVerificationReceiptVerified bool   `json:"ledger_verification_receipt_verified"`
	LedgerVerificationReady           bool   `json:"ledger_verification_ready"`

	WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady         bool   `json:"weighted_admission_resonance_graft_admission_ledger_verification_ready"`
	WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed       bool   `json:"weighted_admission_resonance_graft_admission_ledger_persistence_consumed"`
	WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired       bool   `json:"weighted_admission_resonance_graft_admission_ledger_persistence_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification         bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_verification"`
	WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID            string `json:"weighted_admission_resonance_graft_admission_ledger_verification_id"`
	LedgerVerificationHash                                                  string `json:"ledger_verification_hash"`
	LedgerVerificationReadBackHash                                          string `json:"ledger_verification_read_back_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID       string `json:"source_weighted_admission_resonance_graft_admission_ledger_persistence_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady    bool   `json:"source_weighted_admission_resonance_graft_admission_ledger_persistence_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID string `json:"source_weighted_admission_resonance_graft_admission_ledger_persistence_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash     string `json:"source_weighted_admission_resonance_graft_admission_ledger_persistence_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack string `json:"source_weighted_admission_resonance_graft_admission_ledger_persistence_read_back_hash"`
	SourceLedgerPersistenceReportReceiptShape                               string `json:"source_ledger_persistence_report_receipt_shape"`
	SourceLedgerPersistenceState                                            string `json:"source_ledger_persistence_state"`
	SourceLedgerPersistenceAction                                           string `json:"source_ledger_persistence_action"`
	SourceLedgerPersistenceTarget                                           string `json:"source_ledger_persistence_target"`
	SourceLedgerPersistenceTargetKind                                       string `json:"source_ledger_persistence_target_kind"`
	SourceLedgerPersistenceTargetMode                                       string `json:"source_ledger_persistence_target_mode"`
	SourceLedgerPersistenceReceiptShape                                     string `json:"source_ledger_persistence_receipt_shape"`
	SourceLedgerPersistenceWriteScope                                       string `json:"source_ledger_persistence_write_scope"`
	SourceLedgerPersistenceAppendOnly                                       bool   `json:"source_ledger_persistence_append_only"`
	SourceLedgerPersistenceDryRunOnly                                       bool   `json:"source_ledger_persistence_dry_run_only"`
	SourceLedgerPersistenceReceiptPersisted                                 bool   `json:"source_ledger_persistence_receipt_persisted"`
	SourceLedgerPersistenceReady                                            bool   `json:"source_ledger_persistence_ready"`
	SourceLedgerPersistenceReason                                           string `json:"source_ledger_persistence_reason"`
	SourceLedgerImplementationSchema                                        string `json:"source_ledger_implementation_schema"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerification(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification RESONANCE_GRAFT_ADMISSION_LEDGER_PERSISTENCE_REPORT RESONANCE_GRAFT_ADMISSION_LEDGER_VERIFICATION_REPORT")
	}
	persistencePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification output path missing")
	}
	sourcePersistence, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportForAssert(persistencePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReportError(sourcePersistence, root); err != nil {
		return err
	}
	verification := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport{
		admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReport: sourcePersistence,
		LedgerVerificationState:                                                 "blocked",
		LedgerVerificationAction:                                                "reject_blocked_ledger_persistence",
		LedgerVerificationTarget:                                                "admission_ledger_receipt",
		LedgerVerificationTargetKind:                                            "weighted_internal_world_shadow_graft_admission_ledger_persistence",
		LedgerVerificationTargetMode:                                            "closed_read_back_guard_dry_run",
		LedgerVerificationReceiptShape:                                          "none",
		LedgerVerificationAppendOnly:                                            false,
		LedgerVerificationDryRunOnly:                                            true,
		LedgerVerificationReceiptReadBack:                                       false,
		LedgerVerificationReceiptVerified:                                       false,
		LedgerVerificationReady:                                                 false,
		WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady:         true,
		WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed:       true,
		WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired:       true,
		NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification:         true,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID:       sourcePersistence.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady:    sourcePersistence.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID: sourcePersistence.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash:     sourcePersistence.LedgerPersistenceHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack: sourcePersistence.LedgerPersistenceReadBackHash,
		SourceLedgerPersistenceReportReceiptShape:                               sourcePersistence.ReceiptShape,
		SourceLedgerPersistenceState:                                            sourcePersistence.LedgerPersistenceState,
		SourceLedgerPersistenceAction:                                           sourcePersistence.LedgerPersistenceAction,
		SourceLedgerPersistenceTarget:                                           sourcePersistence.LedgerPersistenceTarget,
		SourceLedgerPersistenceTargetKind:                                       sourcePersistence.LedgerPersistenceTargetKind,
		SourceLedgerPersistenceTargetMode:                                       sourcePersistence.LedgerPersistenceTargetMode,
		SourceLedgerPersistenceReceiptShape:                                     sourcePersistence.LedgerPersistenceReceiptShape,
		SourceLedgerPersistenceWriteScope:                                       sourcePersistence.LedgerPersistenceWriteScope,
		SourceLedgerPersistenceAppendOnly:                                       sourcePersistence.LedgerPersistenceAppendOnly,
		SourceLedgerPersistenceDryRunOnly:                                       sourcePersistence.LedgerPersistenceDryRunOnly,
		SourceLedgerPersistenceReceiptPersisted:                                 sourcePersistence.LedgerPersistenceReceiptPersisted,
		SourceLedgerPersistenceReady:                                            sourcePersistence.LedgerPersistenceReady,
		SourceLedgerPersistenceReason:                                           sourcePersistence.Reason,
		SourceLedgerImplementationSchema:                                        sourcePersistence.SourceSchema,
	}
	verification.Schema = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema
	verification.Status = "shadow_graft_admission_ledger_verification_blocked_dry_run"
	verification.TargetKind = "weighted_internal_world_shadow_graft_admission_ledger_verification"
	verification.TargetMode = "closed_ledger_verification_guard_dry_run"
	verification.Action = "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run"
	verification.WriterAction = "reject_blocked_ledger_persistence"
	verification.RollbackAction = "reject_blocked_ledger_persistence"
	verification.LedgerState = "blocked"
	verification.LedgerAction = "reject_blocked_ledger_persistence"
	verification.LedgerContract = "none"
	verification.LedgerEntrypoint = "none"
	verification.LedgerReceiptShape = "none"
	verification.LedgerWriteScope = "none"
	verification.LedgerReady = false
	verification.LedgerAppendAllowed = false
	verification.ReceiptShape = "weighted_resonance_shadow_graft_admission_ledger_verification_receipt"
	verification.SourceSchema = sourcePersistence.Schema
	verification.SourceStatus = sourcePersistence.Status
	verification.SourceTarget = sourcePersistence.Target
	verification.SourceReport = persistencePath
	verification.Reason = "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed"
	verification.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID(verification)
	verification.LedgerVerificationHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash(verification)
	verification.LedgerVerificationReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBackHash(verification)
	verification.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID(verification)
	if verification.CausalID == "" ||
		verification.LedgerVerificationHash == "" ||
		verification.LedgerVerificationReadBackHash == "" ||
		verification.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID == "" ||
		verification.LedgerVerificationHash == verification.LedgerVerificationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification read-back proof failed")
	}
	raw, err := json.MarshalIndent(verification, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification] pass: resonance_graft_admission_ledger_verification_report=%s resonance_graft_admission_ledger_persistence_report=%s\n", outputPath, persistencePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-ledger-verification-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationSchema)
	}
	if report.Status != "shadow_graft_admission_ledger_verification_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification status mismatch: got %q want %q", report.Status, "shadow_graft_admission_ledger_verification_blocked_dry_run")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_ledger_verification" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_ledger_verification")
	}
	if report.TargetMode != "closed_ledger_verification_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification target_mode mismatch: got %q want %q", report.TargetMode, "closed_ledger_verification_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_ledger_persistence_blocked_dry_run")
	}
	if report.WriterAction != "reject_blocked_ledger_persistence" || report.RollbackAction != "reject_blocked_ledger_persistence" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification writer/rollback action mismatch")
	}
	if report.LedgerState != "blocked" ||
		report.LedgerAction != "reject_blocked_ledger_persistence" ||
		report.LedgerContract != "none" ||
		report.LedgerEntrypoint != "none" ||
		report.LedgerReceiptShape != "none" ||
		report.LedgerWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification ledger guard mismatch")
	}
	if report.LedgerVerificationState != "blocked" ||
		report.LedgerVerificationAction != "reject_blocked_ledger_persistence" ||
		report.LedgerVerificationTarget != "admission_ledger_receipt" ||
		report.LedgerVerificationTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_persistence" ||
		report.LedgerVerificationTargetMode != "closed_read_back_guard_dry_run" ||
		report.LedgerVerificationReceiptShape != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification shape mismatch")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_verification_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_ledger_verification_receipt")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_ledger_verification_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired},
		{"next_step_blocked_without_resonance_graft_admission_ledger_verification", report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_ledger_persistence_ready", report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_consumed", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationConsumed},
		{"weighted_admission_resonance_graft_admission_ledger_implementation_required", report.WeightedAdmissionResonanceGraftAdmissionLedgerImplementationRequired},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger verification %s not ready", required.name)
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
		{"ledger_verification_append_only", report.LedgerVerificationAppendOnly},
		{"ledger_verification_receipt_read_back", report.LedgerVerificationReceiptReadBack},
		{"ledger_verification_receipt_verified", report.LedgerVerificationReceiptVerified},
		{"ledger_verification_ready", report.LedgerVerificationReady},
		{"source_ledger_persistence_append_only", report.SourceLedgerPersistenceAppendOnly},
		{"source_ledger_persistence_receipt_persisted", report.SourceLedgerPersistenceReceiptPersisted},
		{"source_ledger_persistence_ready", report.SourceLedgerPersistenceReady},
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
			return fmt.Errorf("weighted admission resonance graft admission ledger verification opened %s", closed.name)
		}
	}
	if !report.LedgerVerificationDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification ledger_verification_dry_run_only not ready")
	}
	if !report.SourceLedgerPersistenceDryRunOnly {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source_ledger_persistence_dry_run_only not ready")
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_ledger_verification_id", report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID},
		{"causal_id", report.CausalID},
		{"ledger_verification_hash", report.LedgerVerificationHash},
		{"ledger_verification_read_back_hash", report.LedgerVerificationReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash},
		{"source_weighted_admission_resonance_graft_admission_ledger_persistence_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack},
		{"source_ledger_persistence_reason", report.SourceLedgerPersistenceReason},
		{"source_ledger_implementation_schema", report.SourceLedgerImplementationSchema},
		{"source_weighted_admission_resonance_graft_admission_ledger_implementation_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID},
		{"source_weighted_admission_resonance_graft_admission_ledger_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission ledger verification %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_ledger_persistence_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_ledger_persistence_blocked_dry_run")
	}
	if report.SourceLedgerImplementationSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source_ledger_implementation_schema mismatch: got %q want %q", report.SourceLedgerImplementationSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerImplementationSchema)
	}
	if report.SourceLedgerPersistenceReportReceiptShape != "weighted_resonance_shadow_graft_admission_ledger_persistence_receipt" ||
		report.SourceLedgerPersistenceState != "blocked" ||
		report.SourceLedgerPersistenceAction != "reject_blocked_ledger_implementation" ||
		report.SourceLedgerPersistenceTarget != "admission_ledger_receipt" ||
		report.SourceLedgerPersistenceTargetKind != "weighted_internal_world_shadow_graft_admission_ledger_implementation" ||
		report.SourceLedgerPersistenceTargetMode != "closed_persistence_guard_dry_run" ||
		report.SourceLedgerPersistenceReceiptShape != "none" ||
		report.SourceLedgerPersistenceWriteScope != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source ledger persistence shape mismatch")
	}
	if report.SourceLedgerPersistenceReason != "weighted resonance shadow graft admission ledger persistence blocked by blocked ledger implementation; ledger receipt persistence remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source_ledger_persistence_reason mismatch: got %q", report.SourceLedgerPersistenceReason)
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID, "weighted-resonance-graft-admission-ledger-persistence-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceCausalID, "weighted-resonance-graft-admission-ledger-persistence-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash, "weighted-resonance-graft-admission-ledger-persistence-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack, "weighted-resonance-graft-admission-ledger-persistence-read-") {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification source ledger persistence mismatch")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification causal_id mismatch")
	}
	if report.LedgerVerificationHash == "" || report.LedgerVerificationHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification ledger_verification_hash mismatch")
	}
	if report.LedgerVerificationReadBackHash == "" || report.LedgerVerificationReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification ledger_verification_read_back_hash mismatch")
	}
	if report.LedgerVerificationHash == report.LedgerVerificationReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID == "" || report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID(report) {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission ledger verification blocked by blocked ledger persistence; receipt read-back remains closed" {
		return fmt.Errorf("weighted admission resonance graft admission ledger verification reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationCausalID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport) string {
	h := hashJSON(struct {
		SourcePersistenceID   string `json:"source_ledger_persistence_id"`
		SourcePersistenceRead string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementation  string `json:"source_ledger_implementation_id"`
		Target                string `json:"target"`
		State                 string `json:"ledger_verification_state"`
		Action                string `json:"ledger_verification_action"`
	}{
		SourcePersistenceID:   report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		SourcePersistenceRead: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack,
		SourceImplementation:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		Target:                report.Target,
		State:                 report.LedgerVerificationState,
		Action:                report.LedgerVerificationAction,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-verification-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourcePersistenceID    string `json:"source_ledger_persistence_id"`
		SourcePersistenceHash  string `json:"source_ledger_persistence_hash"`
		SourcePersistenceRead  string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementationID string `json:"source_ledger_implementation_id"`
		State                  string `json:"ledger_verification_state"`
		Action                 string `json:"ledger_verification_action"`
		Target                 string `json:"ledger_verification_target"`
		TargetKind             string `json:"ledger_verification_target_kind"`
		TargetMode             string `json:"ledger_verification_target_mode"`
		ReceiptShape           string `json:"ledger_verification_receipt_shape"`
		AppendOnly             bool   `json:"ledger_verification_append_only"`
		DryRunOnly             bool   `json:"ledger_verification_dry_run_only"`
		ReceiptReadBack        bool   `json:"ledger_verification_receipt_read_back"`
		ReceiptVerified        bool   `json:"ledger_verification_receipt_verified"`
		Ready                  bool   `json:"ledger_verification_ready"`
		WeightedReady          bool   `json:"weighted_ledger_verification_ready"`
		SourcePersistenceReady bool   `json:"source_ledger_persistence_ready"`
		LedgerAppendAllowed    bool   `json:"ledger_append_allowed"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		ContractsReady         bool   `json:"contracts_ready"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_verification"`
	}{
		CausalID:               report.CausalID,
		SourcePersistenceID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		SourcePersistenceHash:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash,
		SourcePersistenceRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack,
		SourceImplementationID: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		State:                  report.LedgerVerificationState,
		Action:                 report.LedgerVerificationAction,
		Target:                 report.LedgerVerificationTarget,
		TargetKind:             report.LedgerVerificationTargetKind,
		TargetMode:             report.LedgerVerificationTargetMode,
		ReceiptShape:           report.LedgerVerificationReceiptShape,
		AppendOnly:             report.LedgerVerificationAppendOnly,
		DryRunOnly:             report.LedgerVerificationDryRunOnly,
		ReceiptReadBack:        report.LedgerVerificationReceiptReadBack,
		ReceiptVerified:        report.LedgerVerificationReceiptVerified,
		Ready:                  report.LedgerVerificationReady,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady,
		SourcePersistenceReady: report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReady,
		LedgerAppendAllowed:    report.LedgerAppendAllowed,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
		BodyMutationAllowed:    report.BodyMutationAllowed,
		ContractsReady:         report.ContractsReady,
		NextStepBlockedWithout: report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-verification-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReadBackHash(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport) string {
	h := hashJSON(struct {
		LedgerVerificationHash string `json:"ledger_verification_hash"`
		SourcePersistenceID    string `json:"source_ledger_persistence_id"`
		SourcePersistenceRead  string `json:"source_ledger_persistence_read_back_hash"`
		WeightedReady          bool   `json:"weighted_ledger_verification_ready"`
		PersistenceConsumed    bool   `json:"ledger_persistence_consumed"`
		PersistenceRequired    bool   `json:"ledger_persistence_required"`
		VerificationReady      bool   `json:"ledger_verification_ready"`
		ReceiptVerified        bool   `json:"ledger_verification_receipt_verified"`
		ContractsReady         bool   `json:"contracts_ready"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
	}{
		LedgerVerificationHash: report.LedgerVerificationHash,
		SourcePersistenceID:    report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		SourcePersistenceRead:  report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack,
		WeightedReady:          report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady,
		PersistenceConsumed:    report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceConsumed,
		PersistenceRequired:    report.WeightedAdmissionResonanceGraftAdmissionLedgerPersistenceRequired,
		VerificationReady:      report.LedgerVerificationReady,
		ReceiptVerified:        report.LedgerVerificationReceiptVerified,
		ContractsReady:         report.ContractsReady,
		WriteAllowed:           report.WriteAllowed,
		AdmissionAllowed:       report.AdmissionAllowed,
		LiveAdmissionEnabled:   report.LiveAdmissionEnabled,
		MutatesState:           report.MutatesState,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-verification-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationID(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport) string {
	h := hashJSON(struct {
		Schema                     string `json:"schema"`
		Status                     string `json:"status"`
		Action                     string `json:"action"`
		SourcePersistenceID        string `json:"source_ledger_persistence_id"`
		SourcePersistenceHash      string `json:"source_ledger_persistence_hash"`
		SourcePersistenceRead      string `json:"source_ledger_persistence_read_back_hash"`
		SourceImplementationID     string `json:"source_ledger_implementation_id"`
		CausalID                   string `json:"causal_id"`
		LedgerVerificationHash     string `json:"ledger_verification_hash"`
		LedgerVerificationReadBack string `json:"ledger_verification_read_back_hash"`
		State                      string `json:"ledger_verification_state"`
		ActionVerification         string `json:"ledger_verification_action"`
		Ready                      bool   `json:"weighted_ledger_verification_ready"`
		ReceiptVerified            bool   `json:"ledger_verification_receipt_verified"`
		WriteAllowed               bool   `json:"write_allowed"`
		AdmissionAllowed           bool   `json:"admission_allowed"`
		LiveAdmissionEnabled       bool   `json:"live_admission_enabled"`
		MutatesState               bool   `json:"mutates_state"`
		NextStepBlockedWithout     bool   `json:"next_step_blocked_without_resonance_graft_admission_ledger_verification"`
	}{
		Schema:                     report.Schema,
		Status:                     report.Status,
		Action:                     report.Action,
		SourcePersistenceID:        report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceID,
		SourcePersistenceHash:      report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceHash,
		SourcePersistenceRead:      report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerPersistenceReadBack,
		SourceImplementationID:     report.SourceWeightedAdmissionResonanceGraftAdmissionLedgerImplementationID,
		CausalID:                   report.CausalID,
		LedgerVerificationHash:     report.LedgerVerificationHash,
		LedgerVerificationReadBack: report.LedgerVerificationReadBackHash,
		State:                      report.LedgerVerificationState,
		ActionVerification:         report.LedgerVerificationAction,
		Ready:                      report.WeightedAdmissionResonanceGraftAdmissionLedgerVerificationReady,
		ReceiptVerified:            report.LedgerVerificationReceiptVerified,
		WriteAllowed:               report.WriteAllowed,
		AdmissionAllowed:           report.AdmissionAllowed,
		LiveAdmissionEnabled:       report.LiveAdmissionEnabled,
		MutatesState:               report.MutatesState,
		NextStepBlockedWithout:     report.NextStepBlockedWithoutResonanceGraftAdmissionLedgerVerification,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-ledger-verification-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLedgerVerificationReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger verification path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger verification not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger verification not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger verification JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission ledger verification decode failed: %w", err)
	}
	return report, root, nil
}
