package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedReadinessSchema = "arianna.live_route_weighted_readiness.v1"

type admissionLiveRouteWeightedReadinessReport struct {
	Schema                       string `json:"schema"`
	Status                       string `json:"status"`
	Target                       string `json:"target"`
	BodySmokeWeighted            bool   `json:"body_smoke_weighted"`
	NanoDirectRunner             bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate          bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain      bool   `json:"boundary_report_full_chain"`
	ContractsReady               bool   `json:"contracts_ready"`
	WriteAllowed                 bool   `json:"write_allowed"`
	AdmissionAllowed             bool   `json:"admission_allowed"`
	LiveAdmissionEnabled         bool   `json:"live_admission_enabled"`
	MutatesState                 bool   `json:"mutates_state"`
	BodyWorkdir                  string `json:"body_workdir"`
	BoundaryReport               string `json:"boundary_report"`
	ProofLog                     string `json:"proof_log"`
	FinalGateLog                 string `json:"final_gate_log"`
}

func runAdmissionLiveRouteWeightedReadinessAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-readiness-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedReadinessReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedReadinessReportError(report, root)
}

func admissionLiveRouteWeightedReadinessReportError(report admissionLiveRouteWeightedReadinessReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted readiness report schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedReadinessSchema {
		return fmt.Errorf("weighted readiness report schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedReadinessSchema)
	}
	if report.Status != "ready_closed_dry_run" {
		return fmt.Errorf("weighted readiness report status mismatch: got %q want %q", report.Status, "ready_closed_dry_run")
	}
	if report.Target != "live_admission" {
		return fmt.Errorf("weighted readiness report target mismatch: got %q want %q", report.Target, "live_admission")
	}
	for _, ready := range []struct {
		name  string
		value bool
	}{
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
	} {
		if !ready.value {
			return fmt.Errorf("weighted readiness report %s not ready", ready.name)
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
			return fmt.Errorf("weighted readiness report opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"body_workdir", report.BodyWorkdir},
		{"boundary_report", report.BoundaryReport},
		{"proof_log", report.ProofLog},
		{"final_gate_log", report.FinalGateLog},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted readiness report %s missing", pathField.name)
		}
	}
	return nil
}

func readAdmissionLiveRouteWeightedReadinessReportForAssert(path string) (admissionLiveRouteWeightedReadinessReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedReadinessReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted readiness report path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted readiness report not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted readiness report not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted readiness report JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted readiness report decode failed: %w", err)
	}
	return report, root, nil
}
