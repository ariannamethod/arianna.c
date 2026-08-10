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
