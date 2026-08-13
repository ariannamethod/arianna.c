package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const (
	admissionLiveRouteWeightedAdmissionPermitSchema = "arianna.live_route_weighted_admission_permit.v1"
	weightedAdmissionPermitKey                      = "ARIANNA_WEIGHTED_ADMISSION_PERMIT_DRY_RUN_ONLY"
)

type admissionLiveRouteWeightedAdmissionPermitReport struct {
	Schema                             string `json:"schema"`
	Status                             string `json:"status"`
	Target                             string `json:"target"`
	TargetKind                         string `json:"target_kind"`
	TargetMode                         string `json:"target_mode"`
	Action                             string `json:"action"`
	WeightedAdmissionPermitReady       bool   `json:"weighted_admission_permit_ready"`
	WeightedAdmissionAuthorityConsumed bool   `json:"weighted_admission_authority_consumed"`
	WeightedAdmissionAuthorityRequired bool   `json:"weighted_admission_authority_required"`
	ManualPermitRequested              bool   `json:"manual_permit_requested"`
	PermitKeyMatched                   bool   `json:"permit_key_matched"`
	NextStepBlockedWithoutPermit       bool   `json:"next_step_blocked_without_permit"`
	SourceSchema                       string `json:"source_schema"`
	SourceStatus                       string `json:"source_status"`
	SourceTarget                       string `json:"source_target"`
	SourceReport                       string `json:"source_report"`
	SourceContractReport               string `json:"source_contract_report"`
	SourcePreconditionReport           string `json:"source_precondition_report"`
	SourceReadinessReport              string `json:"source_readiness_report"`
	SourceBodyWorkdir                  string `json:"source_body_workdir"`
	SourceBoundaryReport               string `json:"source_boundary_report"`
	SourceProofLog                     string `json:"source_proof_log"`
	SourceFinalGateLog                 string `json:"source_final_gate_log"`
	BodySmokeWeighted                  bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                   bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof       bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain            bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted             bool   `json:"source_authority_granted"`
	AuthorityGranted                   bool   `json:"authority_granted"`
	ContractsReady                     bool   `json:"contracts_ready"`
	WriteAllowed                       bool   `json:"write_allowed"`
	AdmissionAllowed                   bool   `json:"admission_allowed"`
	LiveAdmissionEnabled               bool   `json:"live_admission_enabled"`
	MutatesState                       bool   `json:"mutates_state"`
	Passed                             bool   `json:"passed"`
	Reason                             string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionPermit(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-permit AUTHORITY_REPORT PERMIT_REPORT")
	}
	authorityPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission permit output path missing")
	}
	authority, root, err := readAdmissionLiveRouteWeightedAdmissionAuthorityReportForAssert(authorityPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionAuthorityReportError(authority, root); err != nil {
		return err
	}
	if os.Getenv("A2A_WEIGHTED_ADMISSION_PERMIT_KEY") != weightedAdmissionPermitKey {
		return fmt.Errorf("A2A_WEIGHTED_ADMISSION_PERMIT_KEY must match dry-run confirmation for weighted admission permit")
	}
	permit := admissionLiveRouteWeightedAdmissionPermitReport{
		Schema:                             admissionLiveRouteWeightedAdmissionPermitSchema,
		Status:                             "operator_permitted_closed_dry_run",
		Target:                             "live_route_admission_permit",
		TargetKind:                         "weighted_live_route_admission_permit",
		TargetMode:                         "permit_closed_dry_run",
		Action:                             "acknowledge_closed_weighted_admission_authority_dry_run",
		WeightedAdmissionPermitReady:       true,
		WeightedAdmissionAuthorityConsumed: true,
		WeightedAdmissionAuthorityRequired: true,
		ManualPermitRequested:              true,
		PermitKeyMatched:                   true,
		NextStepBlockedWithoutPermit:       true,
		SourceSchema:                       authority.Schema,
		SourceStatus:                       authority.Status,
		SourceTarget:                       authority.Target,
		SourceReport:                       authorityPath,
		SourceContractReport:               authority.SourceReport,
		SourcePreconditionReport:           authority.SourcePreconditionReport,
		SourceReadinessReport:              authority.SourceReadinessReport,
		SourceBodyWorkdir:                  authority.SourceBodyWorkdir,
		SourceBoundaryReport:               authority.SourceBoundaryReport,
		SourceProofLog:                     authority.SourceProofLog,
		SourceFinalGateLog:                 authority.SourceFinalGateLog,
		BodySmokeWeighted:                  authority.BodySmokeWeighted,
		NanoDirectRunner:                   authority.NanoDirectRunner,
		NanoDirectFinalGate:                authority.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:       authority.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:            authority.BoundaryReportFullChain,
		SourceAuthorityGranted:             authority.AuthorityGranted,
		AuthorityGranted:                   false,
		ContractsReady:                     false,
		WriteAllowed:                       false,
		AdmissionAllowed:                   false,
		LiveAdmissionEnabled:               false,
		MutatesState:                       false,
		Passed:                             true,
		Reason:                             "operator permit accepted for closed weighted authority; live admission remains disabled",
	}
	raw, err := json.MarshalIndent(permit, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission permit marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission permit write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-permit] pass: permit_report=%s authority_report=%s\n", outputPath, authorityPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionPermitAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-permit-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionPermitReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionPermitReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionPermitReportError(report admissionLiveRouteWeightedAdmissionPermitReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission permit schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionPermitSchema {
		return fmt.Errorf("weighted admission permit schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionPermitSchema)
	}
	if report.Status != "operator_permitted_closed_dry_run" {
		return fmt.Errorf("weighted admission permit status mismatch: got %q want %q", report.Status, "operator_permitted_closed_dry_run")
	}
	if report.Target != "live_route_admission_permit" {
		return fmt.Errorf("weighted admission permit target mismatch: got %q want %q", report.Target, "live_route_admission_permit")
	}
	if report.TargetKind != "weighted_live_route_admission_permit" {
		return fmt.Errorf("weighted admission permit target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_admission_permit")
	}
	if report.TargetMode != "permit_closed_dry_run" {
		return fmt.Errorf("weighted admission permit target_mode mismatch: got %q want %q", report.TargetMode, "permit_closed_dry_run")
	}
	if report.Action != "acknowledge_closed_weighted_admission_authority_dry_run" {
		return fmt.Errorf("weighted admission permit action mismatch: got %q want %q", report.Action, "acknowledge_closed_weighted_admission_authority_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_permit_ready", report.WeightedAdmissionPermitReady},
		{"weighted_admission_authority_consumed", report.WeightedAdmissionAuthorityConsumed},
		{"weighted_admission_authority_required", report.WeightedAdmissionAuthorityRequired},
		{"manual_permit_requested", report.ManualPermitRequested},
		{"permit_key_matched", report.PermitKeyMatched},
		{"next_step_blocked_without_permit", report.NextStepBlockedWithoutPermit},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission permit %s not ready", required.name)
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
			return fmt.Errorf("weighted admission permit opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_contract_report", report.SourceContractReport},
		{"source_precondition_report", report.SourcePreconditionReport},
		{"source_readiness_report", report.SourceReadinessReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission permit %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionAuthoritySchema {
		return fmt.Errorf("weighted admission permit source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionAuthoritySchema)
	}
	if report.SourceStatus != "authority_receipt_closed_dry_run" {
		return fmt.Errorf("weighted admission permit source_status mismatch: got %q want %q", report.SourceStatus, "authority_receipt_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_authority" {
		return fmt.Errorf("weighted admission permit source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_authority")
	}
	if report.Reason != "operator permit accepted for closed weighted authority; live admission remains disabled" {
		return fmt.Errorf("weighted admission permit reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionPermitReportForAssert(path string) (admissionLiveRouteWeightedAdmissionPermitReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionPermitReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission permit path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission permit not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission permit not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission permit JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission permit decode failed: %w", err)
	}
	return report, root, nil
}
