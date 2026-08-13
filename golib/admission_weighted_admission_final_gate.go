package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionFinalGateSchema = "arianna.live_route_weighted_admission_final_gate.v1"

type admissionLiveRouteWeightedAdmissionFinalGateReport struct {
	Schema                                   string `json:"schema"`
	Status                                   string `json:"status"`
	Target                                   string `json:"target"`
	TargetKind                               string `json:"target_kind"`
	TargetMode                               string `json:"target_mode"`
	Action                                   string `json:"action"`
	WeightedAdmissionFinalGateReady          bool   `json:"weighted_admission_final_gate_ready"`
	WeightedAdmissionSealConsumed            bool   `json:"weighted_admission_seal_consumed"`
	WeightedAdmissionSealRequired            bool   `json:"weighted_admission_seal_required"`
	NextStepBlockedWithoutFinalGate          bool   `json:"next_step_blocked_without_final_gate"`
	SourceSchema                             string `json:"source_schema"`
	SourceStatus                             string `json:"source_status"`
	SourceTarget                             string `json:"source_target"`
	SourceReport                             string `json:"source_report"`
	SourcePermitReport                       string `json:"source_permit_report"`
	SourceAuthorityReport                    string `json:"source_authority_report"`
	SourceContractReport                     string `json:"source_contract_report"`
	SourcePreconditionReport                 string `json:"source_precondition_report"`
	SourceReadinessReport                    string `json:"source_readiness_report"`
	SourceBodyWorkdir                        string `json:"source_body_workdir"`
	SourceBoundaryReport                     string `json:"source_boundary_report"`
	SourceProofLog                           string `json:"source_proof_log"`
	SourceFinalGateLog                       string `json:"source_final_gate_log"`
	SourceWeightedAdmissionSealReady         bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed    bool   `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired    bool   `json:"source_weighted_admission_permit_required"`
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

func runAdmissionLiveRouteWeightedAdmissionFinalGate(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-final-gate SEAL_REPORT FINAL_GATE_REPORT")
	}
	sealPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission final gate output path missing")
	}
	seal, root, err := readAdmissionLiveRouteWeightedAdmissionSealReportForAssert(sealPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionSealReportError(seal, root); err != nil {
		return err
	}
	finalGate := admissionLiveRouteWeightedAdmissionFinalGateReport{
		Schema:                                   admissionLiveRouteWeightedAdmissionFinalGateSchema,
		Status:                                   "ready_closed_dry_run",
		Target:                                   "live_route_admission_final_gate",
		TargetKind:                               "weighted_live_route_admission_final_gate",
		TargetMode:                               "final_gate_closed_dry_run",
		Action:                                   "verify_weighted_admission_seal_provenance_dry_run",
		WeightedAdmissionFinalGateReady:          true,
		WeightedAdmissionSealConsumed:            true,
		WeightedAdmissionSealRequired:            true,
		NextStepBlockedWithoutFinalGate:          true,
		SourceSchema:                             seal.Schema,
		SourceStatus:                             seal.Status,
		SourceTarget:                             seal.Target,
		SourceReport:                             sealPath,
		SourcePermitReport:                       seal.SourceReport,
		SourceAuthorityReport:                    seal.SourceAuthorityReport,
		SourceContractReport:                     seal.SourceContractReport,
		SourcePreconditionReport:                 seal.SourcePreconditionReport,
		SourceReadinessReport:                    seal.SourceReadinessReport,
		SourceBodyWorkdir:                        seal.SourceBodyWorkdir,
		SourceBoundaryReport:                     seal.SourceBoundaryReport,
		SourceProofLog:                           seal.SourceProofLog,
		SourceFinalGateLog:                       seal.SourceFinalGateLog,
		SourceWeightedAdmissionSealReady:         seal.WeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:    seal.WeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:    seal.WeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:       seal.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed: seal.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired: seal.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:              seal.SourceManualPermitRequested,
		SourcePermitKeyMatched:                   seal.SourcePermitKeyMatched,
		BodySmokeWeighted:                        seal.BodySmokeWeighted,
		NanoDirectRunner:                         seal.NanoDirectRunner,
		NanoDirectFinalGate:                      seal.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:             seal.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                  seal.BoundaryReportFullChain,
		SourceAuthorityGranted:                   seal.AuthorityGranted,
		AuthorityGranted:                         false,
		ContractsReady:                           false,
		WriteAllowed:                             false,
		AdmissionAllowed:                         false,
		LiveAdmissionEnabled:                     false,
		MutatesState:                             false,
		Passed:                                   true,
		Reason:                                   "weighted admission seal cleared final gate; live admission remains disabled",
	}
	raw, err := json.MarshalIndent(finalGate, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission final gate marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission final gate write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-final-gate] pass: final_gate_report=%s seal_report=%s\n", outputPath, sealPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionFinalGateAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-final-gate-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionFinalGateReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionFinalGateReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionFinalGateReportError(report admissionLiveRouteWeightedAdmissionFinalGateReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission final gate schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionFinalGateSchema {
		return fmt.Errorf("weighted admission final gate schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionFinalGateSchema)
	}
	if report.Status != "ready_closed_dry_run" {
		return fmt.Errorf("weighted admission final gate status mismatch: got %q want %q", report.Status, "ready_closed_dry_run")
	}
	if report.Target != "live_route_admission_final_gate" {
		return fmt.Errorf("weighted admission final gate target mismatch: got %q want %q", report.Target, "live_route_admission_final_gate")
	}
	if report.TargetKind != "weighted_live_route_admission_final_gate" {
		return fmt.Errorf("weighted admission final gate target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_admission_final_gate")
	}
	if report.TargetMode != "final_gate_closed_dry_run" {
		return fmt.Errorf("weighted admission final gate target_mode mismatch: got %q want %q", report.TargetMode, "final_gate_closed_dry_run")
	}
	if report.Action != "verify_weighted_admission_seal_provenance_dry_run" {
		return fmt.Errorf("weighted admission final gate action mismatch: got %q want %q", report.Action, "verify_weighted_admission_seal_provenance_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_final_gate_ready", report.WeightedAdmissionFinalGateReady},
		{"weighted_admission_seal_consumed", report.WeightedAdmissionSealConsumed},
		{"weighted_admission_seal_required", report.WeightedAdmissionSealRequired},
		{"next_step_blocked_without_final_gate", report.NextStepBlockedWithoutFinalGate},
		{"source_weighted_admission_seal_ready", report.SourceWeightedAdmissionSealReady},
		{"source_weighted_admission_permit_consumed", report.SourceWeightedAdmissionPermitConsumed},
		{"source_weighted_admission_permit_required", report.SourceWeightedAdmissionPermitRequired},
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
			return fmt.Errorf("weighted admission final gate %s not ready", required.name)
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
			return fmt.Errorf("weighted admission final gate opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_permit_report", report.SourcePermitReport},
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
			return fmt.Errorf("weighted admission final gate %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionSealSchema {
		return fmt.Errorf("weighted admission final gate source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionSealSchema)
	}
	if report.SourceStatus != "sealed_closed_dry_run" {
		return fmt.Errorf("weighted admission final gate source_status mismatch: got %q want %q", report.SourceStatus, "sealed_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_seal" {
		return fmt.Errorf("weighted admission final gate source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_seal")
	}
	if report.Reason != "weighted admission seal cleared final gate; live admission remains disabled" {
		return fmt.Errorf("weighted admission final gate reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionFinalGateReportForAssert(path string) (admissionLiveRouteWeightedAdmissionFinalGateReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionFinalGateReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission final gate path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission final gate not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission final gate not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission final gate JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission final gate decode failed: %w", err)
	}
	return report, root, nil
}
