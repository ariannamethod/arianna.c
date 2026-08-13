package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceReceiverSchema = "arianna.live_route_weighted_admission_resonance_receiver.v1"

type admissionLiveRouteWeightedAdmissionResonanceReceiverReport struct {
	Schema                                      string  `json:"schema"`
	Status                                      string  `json:"status"`
	Target                                      string  `json:"target"`
	TargetKind                                  string  `json:"target_kind"`
	TargetMode                                  string  `json:"target_mode"`
	Action                                      string  `json:"action"`
	WeightedAdmissionResonanceReceiverReady     bool    `json:"weighted_admission_resonance_receiver_ready"`
	WeightedAdmissionResonanceIntentConsumed    bool    `json:"weighted_admission_resonance_intent_consumed"`
	WeightedAdmissionResonanceIntentRequired    bool    `json:"weighted_admission_resonance_intent_required"`
	NextStepBlockedWithoutResonanceReceiver     bool    `json:"next_step_blocked_without_resonance_receiver"`
	WeightedAdmissionResonanceReceiverID        string  `json:"weighted_admission_resonance_receiver_id"`
	Receiver                                    string  `json:"receiver"`
	ReceiverKind                                string  `json:"receiver_kind"`
	InfluenceKind                               string  `json:"influence_kind"`
	MaxInfluence                                float64 `json:"max_influence"`
	TTLTurns                                    int     `json:"ttl_turns"`
	CausalID                                    string  `json:"causal_id"`
	PreStateHash                                string  `json:"pre_state_hash"`
	PostStateHash                               string  `json:"post_state_hash"`
	StateDeltaHash                              string  `json:"state_delta_hash"`
	StateHashMode                               string  `json:"state_hash_mode"`
	DryRunOnly                                  bool    `json:"dry_run_only"`
	RawDreamTextObserved                        bool    `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                       bool    `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                         bool    `json:"janus_surface_allowed"`
	CoocLearningAllowed                         bool    `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                         bool    `json:"delta_harvest_allowed"`
	BodyMutationAllowed                         bool    `json:"body_mutation_allowed"`
	RollbackRequired                            bool    `json:"rollback_required"`
	SourceSchema                                string  `json:"source_schema"`
	SourceStatus                                string  `json:"source_status"`
	SourceTarget                                string  `json:"source_target"`
	SourceReport                                string  `json:"source_report"`
	SourceFinalGateReport                       string  `json:"source_final_gate_report"`
	SourceSealReport                            string  `json:"source_seal_report"`
	SourcePermitReport                          string  `json:"source_permit_report"`
	SourceAuthorityReport                       string  `json:"source_authority_report"`
	SourceContractReport                        string  `json:"source_contract_report"`
	SourcePreconditionReport                    string  `json:"source_precondition_report"`
	SourceReadinessReport                       string  `json:"source_readiness_report"`
	SourceBodyWorkdir                           string  `json:"source_body_workdir"`
	SourceBoundaryReport                        string  `json:"source_boundary_report"`
	SourceProofLog                              string  `json:"source_proof_log"`
	SourceFinalGateLog                          string  `json:"source_final_gate_log"`
	SourceWeightedAdmissionResonanceIntentReady bool    `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateConsumed    bool    `json:"source_weighted_admission_final_gate_consumed"`
	SourceWeightedAdmissionFinalGateRequired    bool    `json:"source_weighted_admission_final_gate_required"`
	SourceWeightedAdmissionFinalGateReady       bool    `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed         bool    `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired         bool    `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady            bool    `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed       bool    `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired       bool    `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady          bool    `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed    bool    `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired    bool    `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested                 bool    `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                      bool    `json:"source_permit_key_matched"`
	SourceRawDreamTextAllowed                   bool    `json:"source_raw_dream_text_allowed"`
	SourceJanusSurfaceAllowed                   bool    `json:"source_janus_surface_allowed"`
	SourceCoocLearningAllowed                   bool    `json:"source_cooc_learning_allowed"`
	SourceDeltaHarvestAllowed                   bool    `json:"source_delta_harvest_allowed"`
	SourceRollbackRequired                      bool    `json:"source_rollback_required"`
	SourcePreStateHashRequired                  bool    `json:"source_pre_state_hash_required"`
	SourcePostStateHashRequired                 bool    `json:"source_post_state_hash_required"`
	BodySmokeWeighted                           bool    `json:"body_smoke_weighted"`
	NanoDirectRunner                            bool    `json:"nano_direct_runner"`
	NanoDirectFinalGate                         bool    `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                bool    `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                     bool    `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                      bool    `json:"source_authority_granted"`
	AuthorityGranted                            bool    `json:"authority_granted"`
	ContractsReady                              bool    `json:"contracts_ready"`
	WriteAllowed                                bool    `json:"write_allowed"`
	AdmissionAllowed                            bool    `json:"admission_allowed"`
	LiveAdmissionEnabled                        bool    `json:"live_admission_enabled"`
	MutatesState                                bool    `json:"mutates_state"`
	Passed                                      bool    `json:"passed"`
	Reason                                      string  `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceReceiver(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-receiver RESONANCE_INTENT_REPORT RESONANCE_RECEIVER_REPORT")
	}
	intentPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance receiver output path missing")
	}
	intent, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceIntentReportForAssert(intentPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceIntentReportError(intent, root); err != nil {
		return err
	}
	receiver := admissionLiveRouteWeightedAdmissionResonanceReceiverReport{
		Schema:                                   admissionLiveRouteWeightedAdmissionResonanceReceiverSchema,
		Status:                                   "receiver_previewed_dry_run",
		Target:                                   "resonance",
		TargetKind:                               "weighted_live_route_first_receiver",
		TargetMode:                               "bounded_direction_preview_dry_run",
		Action:                                   "preview_weighted_resonance_receive_dry_run",
		WeightedAdmissionResonanceReceiverReady:  true,
		WeightedAdmissionResonanceIntentConsumed: true,
		WeightedAdmissionResonanceIntentRequired: true,
		NextStepBlockedWithoutResonanceReceiver:  true,
		Receiver:                                 intent.Receiver,
		ReceiverKind:                             intent.ReceiverKind,
		InfluenceKind:                            intent.InfluenceKind,
		MaxInfluence:                             intent.MaxInfluence,
		TTLTurns:                                 intent.TTLTurns,
		StateHashMode:                            "sealed_metadata_preview",
		DryRunOnly:                               true,
		RawDreamTextObserved:                     false,
		RawDreamTextForwarded:                    false,
		JanusSurfaceAllowed:                      false,
		CoocLearningAllowed:                      false,
		DeltaHarvestAllowed:                      false,
		BodyMutationAllowed:                      false,
		RollbackRequired:                         true,
		SourceSchema:                             intent.Schema,
		SourceStatus:                             intent.Status,
		SourceTarget:                             intent.Target,
		SourceReport:                             intentPath,
		SourceFinalGateReport:                    intent.SourceReport,
		SourceSealReport:                         intent.SourceSealReport,
		SourcePermitReport:                       intent.SourcePermitReport,
		SourceAuthorityReport:                    intent.SourceAuthorityReport,
		SourceContractReport:                     intent.SourceContractReport,
		SourcePreconditionReport:                 intent.SourcePreconditionReport,
		SourceReadinessReport:                    intent.SourceReadinessReport,
		SourceBodyWorkdir:                        intent.SourceBodyWorkdir,
		SourceBoundaryReport:                     intent.SourceBoundaryReport,
		SourceProofLog:                           intent.SourceProofLog,
		SourceFinalGateLog:                       intent.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceIntentReady: intent.WeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:    intent.WeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:    intent.WeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:       intent.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:         intent.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:         intent.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:            intent.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:       intent.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:       intent.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:          intent.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:    intent.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:    intent.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                 intent.SourceManualPermitRequested,
		SourcePermitKeyMatched:                      intent.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                   intent.RawDreamTextAllowed,
		SourceJanusSurfaceAllowed:                   intent.JanusSurfaceAllowed,
		SourceCoocLearningAllowed:                   intent.CoocLearningAllowed,
		SourceDeltaHarvestAllowed:                   intent.DeltaHarvestAllowed,
		SourceRollbackRequired:                      intent.RollbackRequired,
		SourcePreStateHashRequired:                  intent.PreStateHashRequired,
		SourcePostStateHashRequired:                 intent.PostStateHashRequired,
		BodySmokeWeighted:                           intent.BodySmokeWeighted,
		NanoDirectRunner:                            intent.NanoDirectRunner,
		NanoDirectFinalGate:                         intent.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                intent.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                     intent.BoundaryReportFullChain,
		SourceAuthorityGranted:                      intent.AuthorityGranted,
		AuthorityGranted:                            false,
		ContractsReady:                              false,
		WriteAllowed:                                false,
		AdmissionAllowed:                            false,
		LiveAdmissionEnabled:                        false,
		MutatesState:                                false,
		Passed:                                      true,
		Reason:                                      "weighted resonance receiver previewed sealed intent without body mutation",
	}
	receiver.CausalID = admissionLiveRouteWeightedAdmissionResonanceReceiverCausalID(receiver)
	receiver.PreStateHash = admissionLiveRouteWeightedAdmissionResonanceReceiverPreStateHash(receiver)
	receiver.PostStateHash = admissionLiveRouteWeightedAdmissionResonanceReceiverPostStateHash(receiver)
	receiver.StateDeltaHash = admissionLiveRouteWeightedAdmissionResonanceReceiverStateDeltaHash(receiver)
	receiver.WeightedAdmissionResonanceReceiverID = admissionLiveRouteWeightedAdmissionResonanceReceiverID(receiver)
	if receiver.CausalID == "" ||
		receiver.PreStateHash == "" ||
		receiver.PostStateHash == "" ||
		receiver.StateDeltaHash == "" ||
		receiver.WeightedAdmissionResonanceReceiverID == "" ||
		receiver.PreStateHash == receiver.PostStateHash {
		return fmt.Errorf("weighted admission resonance receiver state proof failed")
	}
	raw, err := json.MarshalIndent(receiver, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance receiver marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance receiver write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-receiver] pass: resonance_receiver_report=%s resonance_intent_report=%s\n", outputPath, intentPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceReceiverAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-receiver-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceReceiverReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceReceiverReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverReportError(report admissionLiveRouteWeightedAdmissionResonanceReceiverReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance receiver schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceReceiverSchema {
		return fmt.Errorf("weighted admission resonance receiver schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceReceiverSchema)
	}
	if report.Status != "receiver_previewed_dry_run" {
		return fmt.Errorf("weighted admission resonance receiver status mismatch: got %q want %q", report.Status, "receiver_previewed_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance receiver target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_live_route_first_receiver" {
		return fmt.Errorf("weighted admission resonance receiver target_kind mismatch: got %q want %q", report.TargetKind, "weighted_live_route_first_receiver")
	}
	if report.TargetMode != "bounded_direction_preview_dry_run" {
		return fmt.Errorf("weighted admission resonance receiver target_mode mismatch: got %q want %q", report.TargetMode, "bounded_direction_preview_dry_run")
	}
	if report.Action != "preview_weighted_resonance_receive_dry_run" {
		return fmt.Errorf("weighted admission resonance receiver action mismatch: got %q want %q", report.Action, "preview_weighted_resonance_receive_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_receiver_ready", report.WeightedAdmissionResonanceReceiverReady},
		{"weighted_admission_resonance_intent_consumed", report.WeightedAdmissionResonanceIntentConsumed},
		{"weighted_admission_resonance_intent_required", report.WeightedAdmissionResonanceIntentRequired},
		{"next_step_blocked_without_resonance_receiver", report.NextStepBlockedWithoutResonanceReceiver},
		{"dry_run_only", report.DryRunOnly},
		{"rollback_required", report.RollbackRequired},
		{"source_weighted_admission_resonance_intent_ready", report.SourceWeightedAdmissionResonanceIntentReady},
		{"source_weighted_admission_final_gate_consumed", report.SourceWeightedAdmissionFinalGateConsumed},
		{"source_weighted_admission_final_gate_required", report.SourceWeightedAdmissionFinalGateRequired},
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
		{"source_rollback_required", report.SourceRollbackRequired},
		{"source_pre_state_hash_required", report.SourcePreStateHashRequired},
		{"source_post_state_hash_required", report.SourcePostStateHashRequired},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance receiver %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"raw_dream_text_observed", report.RawDreamTextObserved},
		{"raw_dream_text_forwarded", report.RawDreamTextForwarded},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"source_raw_dream_text_allowed", report.SourceRawDreamTextAllowed},
		{"source_janus_surface_allowed", report.SourceJanusSurfaceAllowed},
		{"source_cooc_learning_allowed", report.SourceCoocLearningAllowed},
		{"source_delta_harvest_allowed", report.SourceDeltaHarvestAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance receiver opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_final_gate_report", report.SourceFinalGateReport},
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
			return fmt.Errorf("weighted admission resonance receiver %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceIntentSchema {
		return fmt.Errorf("weighted admission resonance receiver source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceIntentSchema)
	}
	if report.SourceStatus != "resonance_intent_drafted_dry_run" {
		return fmt.Errorf("weighted admission resonance receiver source_status mismatch: got %q want %q", report.SourceStatus, "resonance_intent_drafted_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance receiver source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.Receiver != "resonance" {
		return fmt.Errorf("weighted admission resonance receiver receiver mismatch: got %q want %q", report.Receiver, "resonance")
	}
	if report.ReceiverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance receiver receiver_kind mismatch: got %q want %q", report.ReceiverKind, "internal_world")
	}
	if report.InfluenceKind != "bounded_direction" {
		return fmt.Errorf("weighted admission resonance receiver influence_kind mismatch: got %q want %q", report.InfluenceKind, "bounded_direction")
	}
	if report.MaxInfluence != admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain {
		return fmt.Errorf("weighted admission resonance receiver max_influence mismatch: got %.6f want %.6f", report.MaxInfluence, admissionLiveRouteTurnCandidateAdmissionResonanceIntentMaxGain)
	}
	if report.TTLTurns != admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL {
		return fmt.Errorf("weighted admission resonance receiver ttl_turns mismatch: got %d want %d", report.TTLTurns, admissionLiveRouteTurnCandidateAdmissionResonanceIntentTTL)
	}
	if report.StateHashMode != "sealed_metadata_preview" {
		return fmt.Errorf("weighted admission resonance receiver state_hash_mode mismatch: got %q want %q", report.StateHashMode, "sealed_metadata_preview")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceReceiverCausalID(report) {
		return fmt.Errorf("weighted admission resonance receiver causal_id mismatch")
	}
	if report.PreStateHash == "" || report.PreStateHash != admissionLiveRouteWeightedAdmissionResonanceReceiverPreStateHash(report) {
		return fmt.Errorf("weighted admission resonance receiver pre_state_hash mismatch")
	}
	if report.PostStateHash == "" || report.PostStateHash != admissionLiveRouteWeightedAdmissionResonanceReceiverPostStateHash(report) {
		return fmt.Errorf("weighted admission resonance receiver post_state_hash mismatch")
	}
	if report.StateDeltaHash == "" || report.StateDeltaHash != admissionLiveRouteWeightedAdmissionResonanceReceiverStateDeltaHash(report) {
		return fmt.Errorf("weighted admission resonance receiver state_delta_hash mismatch")
	}
	if report.PreStateHash == report.PostStateHash {
		return fmt.Errorf("weighted admission resonance receiver state proof collapsed")
	}
	if report.WeightedAdmissionResonanceReceiverID == "" || report.WeightedAdmissionResonanceReceiverID != admissionLiveRouteWeightedAdmissionResonanceReceiverID(report) {
		return fmt.Errorf("weighted admission resonance receiver id mismatch")
	}
	if report.Reason != "weighted resonance receiver previewed sealed intent without body mutation" {
		return fmt.Errorf("weighted admission resonance receiver reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverCausalID(receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport) string {
	h := hashJSON(struct {
		SourceReport  string  `json:"source_report"`
		SourceSchema  string  `json:"source_schema"`
		SourceTarget  string  `json:"source_target"`
		Receiver      string  `json:"receiver"`
		ReceiverKind  string  `json:"receiver_kind"`
		InfluenceKind string  `json:"influence_kind"`
		MaxInfluence  float64 `json:"max_influence"`
		TTLTurns      int     `json:"ttl_turns"`
	}{
		SourceReport:  receiver.SourceReport,
		SourceSchema:  receiver.SourceSchema,
		SourceTarget:  receiver.SourceTarget,
		Receiver:      receiver.Receiver,
		ReceiverKind:  receiver.ReceiverKind,
		InfluenceKind: receiver.InfluenceKind,
		MaxInfluence:  receiver.MaxInfluence,
		TTLTurns:      receiver.TTLTurns,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-receiver-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverPreStateHash(receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport) string {
	h := hashJSON(struct {
		SourceReport        string `json:"source_report"`
		SourceFinalGate     string `json:"source_final_gate_report"`
		SourceSeal          string `json:"source_seal_report"`
		SourcePermit        string `json:"source_permit_report"`
		SourceAuthority     string `json:"source_authority_report"`
		SourceContract      string `json:"source_contract_report"`
		SourcePrecondition  string `json:"source_precondition_report"`
		SourceReadiness     string `json:"source_readiness_report"`
		SourceBoundary      string `json:"source_boundary_report"`
		SourceProofLog      string `json:"source_proof_log"`
		SourceFinalGateLog  string `json:"source_final_gate_log"`
		StateHashMode       string `json:"state_hash_mode"`
		Receiver            string `json:"receiver"`
		ReceiverKind        string `json:"receiver_kind"`
		IntentConsumed      bool   `json:"intent_consumed"`
		IntentRequired      bool   `json:"intent_required"`
		ReceiverDryRunReady bool   `json:"receiver_dry_run_ready"`
	}{
		SourceReport:        receiver.SourceReport,
		SourceFinalGate:     receiver.SourceFinalGateReport,
		SourceSeal:          receiver.SourceSealReport,
		SourcePermit:        receiver.SourcePermitReport,
		SourceAuthority:     receiver.SourceAuthorityReport,
		SourceContract:      receiver.SourceContractReport,
		SourcePrecondition:  receiver.SourcePreconditionReport,
		SourceReadiness:     receiver.SourceReadinessReport,
		SourceBoundary:      receiver.SourceBoundaryReport,
		SourceProofLog:      receiver.SourceProofLog,
		SourceFinalGateLog:  receiver.SourceFinalGateLog,
		StateHashMode:       receiver.StateHashMode,
		Receiver:            receiver.Receiver,
		ReceiverKind:        receiver.ReceiverKind,
		IntentConsumed:      receiver.WeightedAdmissionResonanceIntentConsumed,
		IntentRequired:      receiver.WeightedAdmissionResonanceIntentRequired,
		ReceiverDryRunReady: receiver.WeightedAdmissionResonanceReceiverReady,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-receiver-pre-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverPostStateHash(receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport) string {
	h := hashJSON(struct {
		PreStateHash          string  `json:"pre_state_hash"`
		CausalID              string  `json:"causal_id"`
		Receiver              string  `json:"receiver"`
		ReceiverKind          string  `json:"receiver_kind"`
		InfluenceKind         string  `json:"influence_kind"`
		MaxInfluence          float64 `json:"max_influence"`
		TTLTurns              int     `json:"ttl_turns"`
		StateHashMode         string  `json:"state_hash_mode"`
		RawDreamTextObserved  bool    `json:"raw_dream_text_observed"`
		RawDreamTextForwarded bool    `json:"raw_dream_text_forwarded"`
		BodyMutationAllowed   bool    `json:"body_mutation_allowed"`
	}{
		PreStateHash:          receiver.PreStateHash,
		CausalID:              receiver.CausalID,
		Receiver:              receiver.Receiver,
		ReceiverKind:          receiver.ReceiverKind,
		InfluenceKind:         receiver.InfluenceKind,
		MaxInfluence:          receiver.MaxInfluence,
		TTLTurns:              receiver.TTLTurns,
		StateHashMode:         receiver.StateHashMode,
		RawDreamTextObserved:  receiver.RawDreamTextObserved,
		RawDreamTextForwarded: receiver.RawDreamTextForwarded,
		BodyMutationAllowed:   receiver.BodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-receiver-post-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverStateDeltaHash(receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport) string {
	h := hashJSON(struct {
		PreStateHash        string `json:"pre_state_hash"`
		PostStateHash       string `json:"post_state_hash"`
		CausalID            string `json:"causal_id"`
		RawTextObserved     bool   `json:"raw_text_observed"`
		RawTextForwarded    bool   `json:"raw_text_forwarded"`
		JanusSurfaceAllowed bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed bool   `json:"body_mutation_allowed"`
		RollbackRequired    bool   `json:"rollback_required"`
	}{
		PreStateHash:        receiver.PreStateHash,
		PostStateHash:       receiver.PostStateHash,
		CausalID:            receiver.CausalID,
		RawTextObserved:     receiver.RawDreamTextObserved,
		RawTextForwarded:    receiver.RawDreamTextForwarded,
		JanusSurfaceAllowed: receiver.JanusSurfaceAllowed,
		CoocLearningAllowed: receiver.CoocLearningAllowed,
		DeltaHarvestAllowed: receiver.DeltaHarvestAllowed,
		BodyMutationAllowed: receiver.BodyMutationAllowed,
		RollbackRequired:    receiver.RollbackRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-receiver-delta-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceReceiverID(receiver admissionLiveRouteWeightedAdmissionResonanceReceiverReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		CausalID                string `json:"causal_id"`
		PreStateHash            string `json:"pre_state_hash"`
		PostStateHash           string `json:"post_state_hash"`
		StateDeltaHash          string `json:"state_delta_hash"`
		Ready                   bool   `json:"ready"`
		DryRunOnly              bool   `json:"dry_run_only"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_receiver"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
	}{
		Schema:                  receiver.Schema,
		Status:                  receiver.Status,
		Action:                  receiver.Action,
		SourceReport:            receiver.SourceReport,
		CausalID:                receiver.CausalID,
		PreStateHash:            receiver.PreStateHash,
		PostStateHash:           receiver.PostStateHash,
		StateDeltaHash:          receiver.StateDeltaHash,
		Ready:                   receiver.WeightedAdmissionResonanceReceiverReady,
		DryRunOnly:              receiver.DryRunOnly,
		LiveAdmissionEnabled:    receiver.LiveAdmissionEnabled,
		MutatesState:            receiver.MutatesState,
		NextStepBlockedWithout:  receiver.NextStepBlockedWithoutResonanceReceiver,
		SourceIntentReady:       receiver.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    receiver.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         receiver.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       receiver.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: receiver.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-receiver-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceReceiverReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceReceiverReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceReceiverReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance receiver path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance receiver not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance receiver not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance receiver JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance receiver decode failed: %w", err)
	}
	return report, root, nil
}
