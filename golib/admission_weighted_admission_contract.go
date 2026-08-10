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
