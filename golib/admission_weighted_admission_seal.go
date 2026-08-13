package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionSealSchema = "arianna.live_route_weighted_admission_seal.v1"

type admissionLiveRouteWeightedAdmissionSealReport struct {
	Schema                                   string `json:"schema"`
	Status                                   string `json:"status"`
	Target                                   string `json:"target"`
	TargetKind                               string `json:"target_kind"`
	TargetMode                               string `json:"target_mode"`
	Action                                   string `json:"action"`
	WeightedAdmissionSealReady               bool   `json:"weighted_admission_seal_ready"`
	WeightedAdmissionPermitConsumed          bool   `json:"weighted_admission_permit_consumed"`
	WeightedAdmissionPermitRequired          bool   `json:"weighted_admission_permit_required"`
	NextStepBlockedWithoutSeal               bool   `json:"next_step_blocked_without_seal"`
	SourceSchema                             string `json:"source_schema"`
	SourceStatus                             string `json:"source_status"`
	SourceTarget                             string `json:"source_target"`
	SourceReport                             string `json:"source_report"`
	SourceAuthorityReport                    string `json:"source_authority_report"`
	SourceContractReport                     string `json:"source_contract_report"`
	SourcePreconditionReport                 string `json:"source_precondition_report"`
	SourceReadinessReport                    string `json:"source_readiness_report"`
	SourceBodyWorkdir                        string `json:"source_body_workdir"`
	SourceBoundaryReport                     string `json:"source_boundary_report"`
	SourceProofLog                           string `json:"source_proof_log"`
	SourceFinalGateLog                       string `json:"source_final_gate_log"`
	SourceWeightedAdmissionPermitReady       bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired bool   `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested              bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                   bool   `json:"source_permit_key_matched"`
	BodySmokeWeighted                        bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                         bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                      bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof             bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                  bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                   bool   `json:"source_authority_granted"`
	AuthorityGranted                         bool   `json:"authority_granted"`
	ContractsReady                           bool   `json:"contracts_ready"`
	WriteAllowed                             bool   `json:"write_allowed"`
	AdmissionAllowed                         bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                     bool   `json:"live_admission_enabled"`
	MutatesState                             bool   `json:"mutates_state"`
	Passed                                   bool   `json:"passed"`
	Reason                                   string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionSeal(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-seal PERMIT_REPORT SEAL_REPORT")
	}
	permitPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission seal output path missing")
	}
	permit, root, err := readAdmissionLiveRouteWeightedAdmissionPermitReportForAssert(permitPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionPermitReportError(permit, root); err != nil {
		return err
	}
	seal := admissionLiveRouteWeightedAdmissionSealReport{
		Schema:                                   admissionLiveRouteWeightedAdmissionSealSchema,
		Status:                                   "sealed_closed_dry_run",
		Target:                                   "live_route_admission_seal",
		TargetKind:                               "weighted_live_route_admission_seal",
		TargetMode:                               "sealed_closed_dry_run",
		Action:                                   "seal_weighted_admission_permit_provenance_dry_run",
		WeightedAdmissionSealReady:               true,
		WeightedAdmissionPermitConsumed:          true,
		WeightedAdmissionPermitRequired:          true,
		NextStepBlockedWithoutSeal:               true,
		SourceSchema:                             permit.Schema,
		SourceStatus:                             permit.Status,
		SourceTarget:                             permit.Target,
		SourceReport:                             permitPath,
		SourceAuthorityReport:                    permit.SourceReport,
		SourceContractReport:                     permit.SourceContractReport,
		SourcePreconditionReport:                 permit.SourcePreconditionReport,
		SourceReadinessReport:                    permit.SourceReadinessReport,
		SourceBodyWorkdir:                        permit.SourceBodyWorkdir,
		SourceBoundaryReport:                     permit.SourceBoundaryReport,
		SourceProofLog:                           permit.SourceProofLog,
		SourceFinalGateLog:                       permit.SourceFinalGateLog,
		SourceWeightedAdmissionPermitReady:       permit.WeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed: permit.WeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired: permit.WeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:              permit.ManualPermitRequested,
		SourcePermitKeyMatched:                   permit.PermitKeyMatched,
		BodySmokeWeighted:                        permit.BodySmokeWeighted,
		NanoDirectRunner:                         permit.NanoDirectRunner,
		NanoDirectFinalGate:                      permit.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:             permit.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                  permit.BoundaryReportFullChain,
		SourceAuthorityGranted:                   permit.AuthorityGranted,
		AuthorityGranted:                         false,
		ContractsReady:                           false,
		WriteAllowed:                             false,
		AdmissionAllowed:                         false,
		LiveAdmissionEnabled:                     false,
		MutatesState:                             false,
		Passed:                                   true,
		Reason:                                   "weighted admission permit sealed as immutable dry-run receipt; live admission remains disabled",
	}
	raw, err := json.MarshalIndent(seal, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission seal marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission seal write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-seal] pass: seal_report=%s permit_report=%s\n", outputPath, permitPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionSealAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-seal-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionSealReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionSealReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionSealReportError(report admissionLiveRouteWeightedAdmissionSealReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission seal schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionSealSchema {
		return fmt.Errorf("weighted admission seal schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionSealSchema)
	}
	if report.Status != "sealed_closed_dry_run" {
		return fmt.Errorf("weighted admission seal status mismatch: got %q want %q", report.Status, "sealed_closed_dry_run")
	}
	if report.Target != "live_route_admission_seal" {
		return fmt.Errorf("weighted admission seal target mismatch: got %q want %q", report.Target, "live_route_admission_seal")
	}
	if report.TargetKind != "weighted_live_route_admission_seal" {
		return fmt.Errorf("weighted admission seal target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_admission_seal")
	}
	if report.TargetMode != "sealed_closed_dry_run" {
		return fmt.Errorf("weighted admission seal target_mode mismatch: got %q want %q", report.TargetMode, "sealed_closed_dry_run")
	}
	if report.Action != "seal_weighted_admission_permit_provenance_dry_run" {
		return fmt.Errorf("weighted admission seal action mismatch: got %q want %q", report.Action, "seal_weighted_admission_permit_provenance_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_seal_ready", report.WeightedAdmissionSealReady},
		{"weighted_admission_permit_consumed", report.WeightedAdmissionPermitConsumed},
		{"weighted_admission_permit_required", report.WeightedAdmissionPermitRequired},
		{"next_step_blocked_without_seal", report.NextStepBlockedWithoutSeal},
		{"source_weighted_admission_permit_ready", report.SourceWeightedAdmissionPermitReady},
		{"source_weighted_admission_authority_consumed", report.SourceWeightedAdmissionAuthorityConsumed},
		{"source_weighted_admission_authority_required", report.SourceWeightedAdmissionAuthorityRequired},
		{"source_manual_permit_requested", report.SourceManualPermitRequested},
		{"source_permit_key_matched", report.SourcePermitKeyMatched},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission seal %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission seal opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_authority_report", report.SourceAuthorityReport},
		{"source_contract_report", report.SourceContractReport},
		{"source_precondition_report", report.SourcePreconditionReport},
		{"source_readiness_report", report.SourceReadinessReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission seal %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionPermitSchema {
		return fmt.Errorf("weighted admission seal source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionPermitSchema)
	}
	if report.SourceStatus != "operator_permitted_closed_dry_run" {
		return fmt.Errorf("weighted admission seal source_status mismatch: got %q want %q", report.SourceStatus, "operator_permitted_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_permit" {
		return fmt.Errorf("weighted admission seal source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_permit")
	}
	if report.Reason != "weighted admission permit sealed as immutable dry-run receipt; live admission remains disabled" {
		return fmt.Errorf("weighted admission seal reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionSealReportForAssert(path string) (admissionLiveRouteWeightedAdmissionSealReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionSealReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission seal path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission seal not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission seal not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission seal JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission seal decode failed: %w", err)
	}
	return report, root, nil
}
