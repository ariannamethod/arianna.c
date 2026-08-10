package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionContractSchema = "arianna.live_route_weighted_admission_contract.v1"

type admissionLiveRouteWeightedAdmissionContractReport struct {
	Schema                                string `json:"schema"`
	Status                                string `json:"status"`
	Target                                string `json:"target"`
	TargetKind                            string `json:"target_kind"`
	TargetMode                            string `json:"target_mode"`
	Action                                string `json:"action"`
	WeightedAdmissionContractReady        bool   `json:"weighted_admission_contract_ready"`
	WeightedReadinessPreconditionConsumed bool   `json:"weighted_readiness_precondition_consumed"`
	WeightedReadinessPreconditionRequired bool   `json:"weighted_readiness_precondition_required"`
	NextStepBlockedWithoutPrecondition    bool   `json:"next_step_blocked_without_precondition"`
	SourceSchema                          string `json:"source_schema"`
	SourceStatus                          string `json:"source_status"`
	SourceTarget                          string `json:"source_target"`
	SourceReport                          string `json:"source_report"`
	SourceReadinessReport                 string `json:"source_readiness_report"`
	SourceBodyWorkdir                     string `json:"source_body_workdir"`
	SourceBoundaryReport                  string `json:"source_boundary_report"`
	SourceProofLog                        string `json:"source_proof_log"`
	SourceFinalGateLog                    string `json:"source_final_gate_log"`
	BodySmokeWeighted                     bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                      bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                   bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof          bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain               bool   `json:"boundary_report_full_chain"`
	ContractsReady                        bool   `json:"contracts_ready"`
	WriteAllowed                          bool   `json:"write_allowed"`
	AdmissionAllowed                      bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                  bool   `json:"live_admission_enabled"`
	MutatesState                          bool   `json:"mutates_state"`
	Passed                                bool   `json:"passed"`
	Reason                                string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionContract(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-contract PRECONDITION_REPORT CONTRACT_REPORT")
	}
	preconditionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission contract output path missing")
	}
	precondition, root, err := readAdmissionLiveRouteWeightedReadinessPreconditionReportForAssert(preconditionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedReadinessPreconditionReportError(precondition, root); err != nil {
		return err
	}
	contract := admissionLiveRouteWeightedAdmissionContractReport{
		Schema:                                admissionLiveRouteWeightedAdmissionContractSchema,
		Status:                                "contract_ready_closed_dry_run",
		Target:                                "live_route_admission",
		TargetKind:                            "weighted_live_route_admission_contract",
		TargetMode:                            "closed_contract_dry_run",
		Action:                                "bind_weighted_precondition_before_live_route_admission",
		WeightedAdmissionContractReady:        true,
		WeightedReadinessPreconditionConsumed: true,
		WeightedReadinessPreconditionRequired: true,
		NextStepBlockedWithoutPrecondition:    true,
		SourceSchema:                          precondition.Schema,
		SourceStatus:                          precondition.Status,
		SourceTarget:                          precondition.Target,
		SourceReport:                          preconditionPath,
		SourceReadinessReport:                 precondition.SourceReport,
		SourceBodyWorkdir:                     precondition.SourceBodyWorkdir,
		SourceBoundaryReport:                  precondition.SourceBoundaryReport,
		SourceProofLog:                        precondition.SourceProofLog,
		SourceFinalGateLog:                    precondition.SourceFinalGateLog,
		BodySmokeWeighted:                     precondition.BodySmokeWeighted,
		NanoDirectRunner:                      precondition.NanoDirectRunner,
		NanoDirectFinalGate:                   precondition.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:          precondition.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:               precondition.BoundaryReportFullChain,
		ContractsReady:                        false,
		WriteAllowed:                          false,
		AdmissionAllowed:                      false,
		LiveAdmissionEnabled:                  false,
		MutatesState:                          false,
		Passed:                                true,
		Reason:                                "weighted readiness precondition bound; live route admission remains disabled",
	}
	raw, err := json.MarshalIndent(contract, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission contract marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission contract write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-contract] pass: contract_report=%s precondition_report=%s\n", outputPath, preconditionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionContractAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-contract-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionContractReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionContractReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionContractReportError(report admissionLiveRouteWeightedAdmissionContractReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission contract schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionContractSchema {
		return fmt.Errorf("weighted admission contract schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionContractSchema)
	}
	if report.Status != "contract_ready_closed_dry_run" {
		return fmt.Errorf("weighted admission contract status mismatch: got %q want %q", report.Status, "contract_ready_closed_dry_run")
	}
	if report.Target != "live_route_admission" {
		return fmt.Errorf("weighted admission contract target mismatch: got %q want %q", report.Target, "live_route_admission")
	}
	if report.TargetKind != "weighted_live_route_admission_contract" {
		return fmt.Errorf("weighted admission contract target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_admission_contract")
	}
	if report.TargetMode != "closed_contract_dry_run" {
		return fmt.Errorf("weighted admission contract target_mode mismatch: got %q want %q", report.TargetMode, "closed_contract_dry_run")
	}
	if report.Action != "bind_weighted_precondition_before_live_route_admission" {
		return fmt.Errorf("weighted admission contract action mismatch: got %q want %q", report.Action, "bind_weighted_precondition_before_live_route_admission")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_contract_ready", report.WeightedAdmissionContractReady},
		{"weighted_readiness_precondition_consumed", report.WeightedReadinessPreconditionConsumed},
		{"weighted_readiness_precondition_required", report.WeightedReadinessPreconditionRequired},
		{"next_step_blocked_without_precondition", report.NextStepBlockedWithoutPrecondition},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission contract %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission contract opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_readiness_report", report.SourceReadinessReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission contract %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedReadinessPreconditionSchema {
		return fmt.Errorf("weighted admission contract source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedReadinessPreconditionSchema)
	}
	if report.SourceStatus != "precondition_satisfied_closed_dry_run" {
		return fmt.Errorf("weighted admission contract source_status mismatch: got %q want %q", report.SourceStatus, "precondition_satisfied_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission contract source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.Reason != "weighted readiness precondition bound; live route admission remains disabled" {
		return fmt.Errorf("weighted admission contract reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionContractReportForAssert(path string) (admissionLiveRouteWeightedAdmissionContractReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionContractReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission contract path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission contract not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission contract not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission contract JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission contract decode failed: %w", err)
	}
	return report, root, nil
}
