package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedReadinessPreconditionSchema = "arianna.live_route_weighted_readiness_precondition.v1"

type admissionLiveRouteWeightedReadinessPreconditionReport struct {
	Schema                          string `json:"schema"`
	Status                          string `json:"status"`
	Target                          string `json:"target"`
	TargetKind                      string `json:"target_kind"`
	TargetMode                      string `json:"target_mode"`
	Action                          string `json:"action"`
	WeightedReadinessConsumed       bool   `json:"weighted_readiness_consumed"`
	WeightedReadinessRequired       bool   `json:"weighted_readiness_required"`
	NextStepBlockedWithoutReadiness bool   `json:"next_step_blocked_without_readiness"`
	SourceSchema                    string `json:"source_schema"`
	SourceStatus                    string `json:"source_status"`
	SourceTarget                    string `json:"source_target"`
	SourceReport                    string `json:"source_report"`
	SourceBodyWorkdir               string `json:"source_body_workdir"`
	SourceBoundaryReport            string `json:"source_boundary_report"`
	SourceProofLog                  string `json:"source_proof_log"`
	SourceFinalGateLog              string `json:"source_final_gate_log"`
	BodySmokeWeighted               bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate             bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof    bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain         bool   `json:"boundary_report_full_chain"`
	ContractsReady                  bool   `json:"contracts_ready"`
	WriteAllowed                    bool   `json:"write_allowed"`
	AdmissionAllowed                bool   `json:"admission_allowed"`
	LiveAdmissionEnabled            bool   `json:"live_admission_enabled"`
	MutatesState                    bool   `json:"mutates_state"`
	Passed                          bool   `json:"passed"`
	Reason                          string `json:"reason"`
}

func runAdmissionLiveRouteWeightedReadinessPrecondition(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-readiness-precondition READINESS_REPORT PRECONDITION_REPORT")
	}
	readinessPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted readiness precondition output path missing")
	}
	readiness, root, err := readAdmissionLiveRouteWeightedReadinessReportForAssert(readinessPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedReadinessReportError(readiness, root); err != nil {
		return err
	}
	precondition := admissionLiveRouteWeightedReadinessPreconditionReport{
		Schema:                          admissionLiveRouteWeightedReadinessPreconditionSchema,
		Status:                          "precondition_satisfied_closed_dry_run",
		Target:                          "live_route_admission_next_step",
		TargetKind:                      "weighted_pre_live_admission_precondition",
		TargetMode:                      "closed_dry_run",
		Action:                          "consume_weighted_readiness_before_live_route_admission",
		WeightedReadinessConsumed:       true,
		WeightedReadinessRequired:       true,
		NextStepBlockedWithoutReadiness: true,
		SourceSchema:                    readiness.Schema,
		SourceStatus:                    readiness.Status,
		SourceTarget:                    readiness.Target,
		SourceReport:                    readinessPath,
		SourceBodyWorkdir:               readiness.BodyWorkdir,
		SourceBoundaryReport:            readiness.BoundaryReport,
		SourceProofLog:                  readiness.ProofLog,
		SourceFinalGateLog:              readiness.FinalGateLog,
		BodySmokeWeighted:               readiness.BodySmokeWeighted,
		NanoDirectRunner:                readiness.NanoDirectRunner,
		NanoDirectFinalGate:             readiness.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:    readiness.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:         readiness.BoundaryReportFullChain,
		ContractsReady:                  false,
		WriteAllowed:                    false,
		AdmissionAllowed:                false,
		LiveAdmissionEnabled:            false,
		MutatesState:                    false,
		Passed:                          true,
		Reason:                          "weighted readiness consumed; live route admission remains disabled",
	}
	raw, err := json.MarshalIndent(precondition, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted readiness precondition marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted readiness precondition write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-readiness-precondition] pass: precondition_report=%s readiness_report=%s\n", outputPath, readinessPath)
	return nil
}

func admissionLiveRouteWeightedReadinessPreconditionReportError(report admissionLiveRouteWeightedReadinessPreconditionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted readiness precondition schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedReadinessPreconditionSchema {
		return fmt.Errorf("weighted readiness precondition schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedReadinessPreconditionSchema)
	}
	if report.Status != "precondition_satisfied_closed_dry_run" {
		return fmt.Errorf("weighted readiness precondition status mismatch: got %q want %q", report.Status, "precondition_satisfied_closed_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted readiness precondition target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_pre_live_admission_precondition" {
		return fmt.Errorf("weighted readiness precondition target_kind mismatch: got %q want %q", report.TargetKind, "weighted_pre_live_admission_precondition")
	}
	if report.TargetMode != "closed_dry_run" {
		return fmt.Errorf("weighted readiness precondition target_mode mismatch: got %q want %q", report.TargetMode, "closed_dry_run")
	}
	if report.Action != "consume_weighted_readiness_before_live_route_admission" {
		return fmt.Errorf("weighted readiness precondition action mismatch: got %q want %q", report.Action, "consume_weighted_readiness_before_live_route_admission")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_readiness_consumed", report.WeightedReadinessConsumed},
		{"weighted_readiness_required", report.WeightedReadinessRequired},
		{"next_step_blocked_without_readiness", report.NextStepBlockedWithoutReadiness},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted readiness precondition %s not ready", required.name)
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
			return fmt.Errorf("weighted readiness precondition opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_body_workdir", report.SourceBodyWorkdir},
		{"source_boundary_report", report.SourceBoundaryReport},
		{"source_proof_log", report.SourceProofLog},
		{"source_final_gate_log", report.SourceFinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted readiness precondition %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedReadinessSchema {
		return fmt.Errorf("weighted readiness precondition source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedReadinessSchema)
	}
	if report.SourceStatus != "ready_closed_dry_run" {
		return fmt.Errorf("weighted readiness precondition source_status mismatch: got %q want %q", report.SourceStatus, "ready_closed_dry_run")
	}
	if report.SourceTarget != "live_admission" {
		return fmt.Errorf("weighted readiness precondition source_target mismatch: got %q want %q", report.SourceTarget, "live_admission")
	}
	if report.Reason != "weighted readiness consumed; live route admission remains disabled" {
		return fmt.Errorf("weighted readiness precondition reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedReadinessPreconditionReportForAssert(path string) (admissionLiveRouteWeightedReadinessPreconditionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedReadinessPreconditionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted readiness precondition path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted readiness precondition not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted readiness precondition not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted readiness precondition JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted readiness precondition decode failed: %w", err)
	}
	return report, root, nil
}
