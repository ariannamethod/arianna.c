package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionAuthoritySchema = "arianna.live_route_weighted_admission_authority.v1"

type admissionLiveRouteWeightedAdmissionAuthorityReport struct {
	Schema                                 string `json:"schema"`
	Status                                 string `json:"status"`
	Target                                 string `json:"target"`
	TargetKind                             string `json:"target_kind"`
	TargetMode                             string `json:"target_mode"`
	Action                                 string `json:"action"`
	WeightedAdmissionAuthorityReceiptReady bool   `json:"weighted_admission_authority_receipt_ready"`
	WeightedAdmissionContractConsumed      bool   `json:"weighted_admission_contract_consumed"`
	WeightedAdmissionContractRequired      bool   `json:"weighted_admission_contract_required"`
	NextStepBlockedWithoutAuthority        bool   `json:"next_step_blocked_without_authority"`
	SourceSchema                           string `json:"source_schema"`
	SourceStatus                           string `json:"source_status"`
	SourceTarget                           string `json:"source_target"`
	SourceReport                           string `json:"source_report"`
	SourcePreconditionReport               string `json:"source_precondition_report"`
	SourceReadinessReport                  string `json:"source_readiness_report"`
	SourceBodyWorkdir                      string `json:"source_body_workdir"`
	SourceBoundaryReport                   string `json:"source_boundary_report"`
	SourceProofLog                         string `json:"source_proof_log"`
	SourceFinalGateLog                     string `json:"source_final_gate_log"`
	BodySmokeWeighted                      bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                       bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                    bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof           bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                bool   `json:"boundary_report_full_chain"`
	AuthorityGranted                       bool   `json:"authority_granted"`
	ContractsReady                         bool   `json:"contracts_ready"`
	WriteAllowed                           bool   `json:"write_allowed"`
	AdmissionAllowed                       bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                   bool   `json:"live_admission_enabled"`
	MutatesState                           bool   `json:"mutates_state"`
	Passed                                 bool   `json:"passed"`
	Reason                                 string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionAuthority(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-authority CONTRACT_REPORT AUTHORITY_REPORT")
	}
	contractPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission authority output path missing")
	}
	contract, root, err := readAdmissionLiveRouteWeightedAdmissionContractReportForAssert(contractPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionContractReportError(contract, root); err != nil {
		return err
	}
	authority := admissionLiveRouteWeightedAdmissionAuthorityReport{
		Schema:                                 admissionLiveRouteWeightedAdmissionAuthoritySchema,
		Status:                                 "authority_receipt_closed_dry_run",
		Target:                                 "live_route_admission_authority",
		TargetKind:                             "weighted_live_route_admission_authority",
		TargetMode:                             "closed_authority_dry_run",
		Action:                                 "consume_weighted_admission_contract_before_live_authority",
		WeightedAdmissionAuthorityReceiptReady: true,
		WeightedAdmissionContractConsumed:      true,
		WeightedAdmissionContractRequired:      true,
		NextStepBlockedWithoutAuthority:        true,
		SourceSchema:                           contract.Schema,
		SourceStatus:                           contract.Status,
		SourceTarget:                           contract.Target,
		SourceReport:                           contractPath,
		SourcePreconditionReport:               contract.SourceReport,
		SourceReadinessReport:                  contract.SourceReadinessReport,
		SourceBodyWorkdir:                      contract.SourceBodyWorkdir,
		SourceBoundaryReport:                   contract.SourceBoundaryReport,
		SourceProofLog:                         contract.SourceProofLog,
		SourceFinalGateLog:                     contract.SourceFinalGateLog,
		BodySmokeWeighted:                      contract.BodySmokeWeighted,
		NanoDirectRunner:                       contract.NanoDirectRunner,
		NanoDirectFinalGate:                    contract.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:           contract.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                contract.BoundaryReportFullChain,
		AuthorityGranted:                       false,
		ContractsReady:                         false,
		WriteAllowed:                           false,
		AdmissionAllowed:                       false,
		LiveAdmissionEnabled:                   false,
		MutatesState:                           false,
		Passed:                                 true,
		Reason:                                 "weighted admission contract consumed; live authority remains disabled",
	}
	raw, err := json.MarshalIndent(authority, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission authority marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission authority write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-authority] pass: authority_report=%s contract_report=%s\n", outputPath, contractPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionAuthorityAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-authority-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionAuthorityReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionAuthorityReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionAuthorityReportError(report admissionLiveRouteWeightedAdmissionAuthorityReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission authority schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionAuthoritySchema {
		return fmt.Errorf("weighted admission authority schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionAuthoritySchema)
	}
	if report.Status != "authority_receipt_closed_dry_run" {
		return fmt.Errorf("weighted admission authority status mismatch: got %q want %q", report.Status, "authority_receipt_closed_dry_run")
	}
	if report.Target != "live_route_admission_authority" {
		return fmt.Errorf("weighted admission authority target mismatch: got %q want %q", report.Target, "live_route_admission_authority")
	}
	if report.TargetKind != "weighted_live_route_admission_authority" {
		return fmt.Errorf("weighted admission authority target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_admission_authority")
	}
	if report.TargetMode != "closed_authority_dry_run" {
		return fmt.Errorf("weighted admission authority target_mode mismatch: got %q want %q", report.TargetMode, "closed_authority_dry_run")
	}
	if report.Action != "consume_weighted_admission_contract_before_live_authority" {
		return fmt.Errorf("weighted admission authority action mismatch: got %q want %q", report.Action, "consume_weighted_admission_contract_before_live_authority")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_authority_receipt_ready", report.WeightedAdmissionAuthorityReceiptReady},
		{"weighted_admission_contract_consumed", report.WeightedAdmissionContractConsumed},
		{"weighted_admission_contract_required", report.WeightedAdmissionContractRequired},
		{"next_step_blocked_without_authority", report.NextStepBlockedWithoutAuthority},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission authority %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission authority opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_precondition_report", report.SourcePreconditionReport},
		{"source_readiness_report", report.SourceReadinessReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission authority %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionContractSchema {
		return fmt.Errorf("weighted admission authority source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionContractSchema)
	}
	if report.SourceStatus != "contract_ready_closed_dry_run" {
		return fmt.Errorf("weighted admission authority source_status mismatch: got %q want %q", report.SourceStatus, "contract_ready_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission" {
		return fmt.Errorf("weighted admission authority source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission")
	}
	if report.Reason != "weighted admission contract consumed; live authority remains disabled" {
		return fmt.Errorf("weighted admission authority reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionAuthorityReportForAssert(path string) (admissionLiveRouteWeightedAdmissionAuthorityReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionAuthorityReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission authority path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission authority not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission authority not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission authority JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission authority decode failed: %w", err)
	}
	return report, root, nil
}
