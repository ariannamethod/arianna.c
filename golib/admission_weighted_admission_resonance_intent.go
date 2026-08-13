package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceIntentSchema = "arianna.live_route_weighted_admission_resonance_intent.v1"

type admissionLiveRouteWeightedAdmissionResonanceIntentReport struct {
	Schema                                   string  `json:"schema"`
	Status                                   string  `json:"status"`
	Target                                   string  `json:"target"`
	TargetKind                               string  `json:"target_kind"`
	TargetMode                               string  `json:"target_mode"`
	Action                                   string  `json:"action"`
	WeightedAdmissionResonanceIntentReady    bool    `json:"weighted_admission_resonance_intent_ready"`
	WeightedAdmissionFinalGateConsumed       bool    `json:"weighted_admission_final_gate_consumed"`
	WeightedAdmissionFinalGateRequired       bool    `json:"weighted_admission_final_gate_required"`
	NextStepBlockedWithoutResonanceIntent    bool    `json:"next_step_blocked_without_resonance_intent"`
	Receiver                                 string  `json:"receiver"`
	ReceiverKind                             string  `json:"receiver_kind"`
	InfluenceKind                            string  `json:"influence_kind"`
	MaxInfluence                             float64 `json:"max_influence"`
	TTLTurns                                 int     `json:"ttl_turns"`
	RawDreamTextAllowed                      bool    `json:"raw_dream_text_allowed"`
	JanusSurfaceAllowed                      bool    `json:"janus_surface_allowed"`
	CoocLearningAllowed                      bool    `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                      bool    `json:"delta_harvest_allowed"`
	RollbackRequired                         bool    `json:"rollback_required"`
	PreStateHashRequired                     bool    `json:"pre_state_hash_required"`
	PostStateHashRequired                    bool    `json:"post_state_hash_required"`
	SourceSchema                             string  `json:"source_schema"`
	SourceStatus                             string  `json:"source_status"`
	SourceTarget                             string  `json:"source_target"`
	SourceReport                             string  `json:"source_report"`
	SourceSealReport                         string  `json:"source_seal_report"`
	SourcePermitReport                       string  `json:"source_permit_report"`
	SourceAuthorityReport                    string  `json:"source_authority_report"`
	SourceContractReport                     string  `json:"source_contract_report"`
	SourcePreconditionReport                 string  `json:"source_precondition_report"`
	SourceReadinessReport                    string  `json:"source_readiness_report"`
	SourceBodyWorkdir                        string  `json:"source_body_workdir"`
	SourceBoundaryReport                     string  `json:"source_boundary_report"`
	SourceProofLog                           string  `json:"source_proof_log"`
	SourceFinalGateLog                       string  `json:"source_final_gate_log"`
	SourceWeightedAdmissionFinalGateReady    bool    `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed      bool    `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired      bool    `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady         bool    `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed    bool    `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired    bool    `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady       bool    `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed bool    `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired bool    `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested              bool    `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                   bool    `json:"source_permit_key_matched"`
	BodySmokeWeighted                        bool    `json:"body_smoke_weighted"`
	NanoDirectRunner                         bool    `json:"nano_direct_runner"`
	NanoDirectFinalGate                      bool    `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof             bool    `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                  bool    `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                   bool    `json:"source_authority_granted"`
	AuthorityGranted                         bool    `json:"authority_granted"`
	ContractsReady                           bool    `json:"contracts_ready"`
	WriteAllowed                             bool    `json:"write_allowed"`
	AdmissionAllowed                         bool    `json:"admission_allowed"`
	LiveAdmissionEnabled                     bool    `json:"live_admission_enabled"`
	MutatesState                             bool    `json:"mutates_state"`
	Passed                                   bool    `json:"passed"`
	Reason                                   string  `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceIntent(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-intent FINAL_GATE_REPORT RESONANCE_INTENT_REPORT")
	}
	finalGatePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance intent output path missing")
	}
	finalGate, root, err := readAdmissionLiveRouteWeightedAdmissionFinalGateReportForAssert(finalGatePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionFinalGateReportError(finalGate, root); err != nil {
		return err
	}
	intent := admissionLiveRouteWeightedAdmissionResonanceIntentReport{
		Schema:                                   admissionLiveRouteWeightedAdmissionResonanceIntentSchema,
		Status:                                   "resonance_intent_drafted_dry_run",
		Target:                                   "resonance",
		TargetKind:                               "weighted_live_route_first_receiver",
		TargetMode:                               "bounded_direction_dry_run",
		Action:                                   "draft_weighted_resonance_direction_intent_dry_run",
		WeightedAdmissionResonanceIntentReady:    true,
		WeightedAdmissionFinalGateConsumed:       true,
		WeightedAdmissionFinalGateRequired:       true,
		NextStepBlockedWithoutResonanceIntent:    true,
		Receiver:                                 "resonance",
		ReceiverKind:                             "internal_world",
		InfluenceKind:                            "bounded_direction",
		MaxInfluence:                             admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain,
		TTLTurns:                                 admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL,
		RawDreamTextAllowed:                      false,
		JanusSurfaceAllowed:                      false,
		CoocLearningAllowed:                      false,
		DeltaHarvestAllowed:                      false,
		RollbackRequired:                         true,
		PreStateHashRequired:                     true,
		PostStateHashRequired:                    true,
		SourceSchema:                             finalGate.Schema,
		SourceStatus:                             finalGate.Status,
		SourceTarget:                             finalGate.Target,
		SourceReport:                             finalGatePath,
		SourceSealReport:                         finalGate.SourceReport,
		SourcePermitReport:                       finalGate.SourcePermitReport,
		SourceAuthorityReport:                    finalGate.SourceAuthorityReport,
		SourceContractReport:                     finalGate.SourceContractReport,
		SourcePreconditionReport:                 finalGate.SourcePreconditionReport,
		SourceReadinessReport:                    finalGate.SourceReadinessReport,
		SourceBodyWorkdir:                        finalGate.SourceBodyWorkdir,
		SourceBoundaryReport:                     finalGate.SourceBoundaryReport,
		SourceProofLog:                           finalGate.SourceProofLog,
		SourceFinalGateLog:                       finalGate.SourceFinalGateLog,
		SourceWeightedAdmissionFinalGateReady:    finalGate.WeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:      finalGate.WeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:      finalGate.WeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:         finalGate.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:    finalGate.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:    finalGate.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:       finalGate.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed: finalGate.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired: finalGate.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:              finalGate.SourceManualPermitRequested,
		SourcePermitKeyMatched:                   finalGate.SourcePermitKeyMatched,
		BodySmokeWeighted:                        finalGate.BodySmokeWeighted,
		NanoDirectRunner:                         finalGate.NanoDirectRunner,
		NanoDirectFinalGate:                      finalGate.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:             finalGate.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                  finalGate.BoundaryReportFullChain,
		SourceAuthorityGranted:                   finalGate.AuthorityGranted,
		AuthorityGranted:                         false,
		ContractsReady:                           false,
		WriteAllowed:                             false,
		AdmissionAllowed:                         false,
		LiveAdmissionEnabled:                     false,
		MutatesState:                             false,
		Passed:                                   true,
		Reason:                                   "weighted resonance intent drafted from final gate; live admission remains disabled",
	}
	raw, err := json.MarshalIndent(intent, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance intent marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance intent write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-intent] pass: resonance_intent_report=%s final_gate_report=%s\n", outputPath, finalGatePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceIntentAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-intent-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceIntentReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceIntentReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceIntentReportError(report admissionLiveRouteWeightedAdmissionResonanceIntentReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance intent schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceIntentSchema {
		return fmt.Errorf("weighted admission resonance intent schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceIntentSchema)
	}
	if report.Status != "resonance_intent_drafted_dry_run" {
		return fmt.Errorf("weighted admission resonance intent status mismatch: got %q want %q", report.Status, "resonance_intent_drafted_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance intent target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_live_route_first_receiver" {
		return fmt.Errorf("weighted admission resonance intent target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_first_receiver")
	}
	if report.TargetMode != "bounded_direction_dry_run" {
		return fmt.Errorf("weighted admission resonance intent target_mode mismatch: got %q want %q", report.TargetMode, "bounded_direction_dry_run")
	}
	if report.Action != "draft_weighted_resonance_direction_intent_dry_run" {
		return fmt.Errorf("weighted admission resonance intent action mismatch: got %q want %q", report.Action, "draft_weighted_resonance_direction_intent_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_intent_ready", report.WeightedAdmissionResonanceIntentReady},
		{"weighted_admission_final_gate_consumed", report.WeightedAdmissionFinalGateConsumed},
		{"weighted_admission_final_gate_required", report.WeightedAdmissionFinalGateRequired},
		{"next_step_blocked_without_resonance_intent", report.NextStepBlockedWithoutResonanceIntent},
		{"rollback_required", report.RollbackRequired},
		{"pre_state_hash_required", report.PreStateHashRequired},
		{"post_state_hash_required", report.PostStateHashRequired},
		{"source_weighted_admission_final_gate_ready", report.SourceWeightedAdmissionFinalGateReady},
		{"source_weighted_admission_seal_consumed", report.SourceWeightedAdmissionSealConsumed},
		{"source_weighted_admission_seal_required", report.SourceWeightedAdmissionSealRequired},
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
			return fmt.Errorf("weighted admission resonance intent %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance intent opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_seal_report", report.SourceSealReport},
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
			return fmt.Errorf("weighted admission resonance intent %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionFinalGateSchema {
		return fmt.Errorf("weighted admission resonance intent source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionFinalGateSchema)
	}
	if report.SourceStatus != "ready_closed_dry_run" {
		return fmt.Errorf("weighted admission resonance intent source_status mismatch: got %q want %q", report.SourceStatus, "ready_closed_dry_run")
	}
	if report.SourceTarget != "live_route_admission_final_gate" {
		return fmt.Errorf("weighted admission resonance intent source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_final_gate")
	}
	if report.Receiver != "resonance" {
		return fmt.Errorf("weighted admission resonance intent receiver mismatch: got %q want %q", report.Receiver, "resonance")
	}
	if report.ReceiverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance intent receiver_kind mismatch: got %q want %q", report.ReceiverKind, "internal_world")
	}
	if report.InfluenceKind != "bounded_direction" {
		return fmt.Errorf("weighted admission resonance intent influence_kind mismatch: got %q want %q", report.InfluenceKind, "bounded_direction")
	}
	if report.MaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain {
		return fmt.Errorf("weighted admission resonance intent max_influence mismatch: got %.6f want %.6f", report.MaxInfluence, admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain)
	}
	if report.TTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL {
		return fmt.Errorf("weighted admission resonance intent ttl_turns mismatch: got %d want %d", report.TTLTurns, admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL)
	}
	if report.Reason != "weighted resonance intent drafted from final gate; live admission remains disabled" {
		return fmt.Errorf("weighted admission resonance intent reason mismatch: got %q", report.Reason)
	}
	return nil
}

func readAdmissionLiveRouteWeightedAdmissionResonanceIntentReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceIntentReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceIntentReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance intent path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance intent not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance intent not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance intent JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance intent decode failed: %w", err)
	}
	return report, root, nil
}
