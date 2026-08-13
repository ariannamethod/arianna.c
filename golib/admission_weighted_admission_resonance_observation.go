package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceObservationSchema = "arianna.live_route_weighted_admission_resonance_observation.v1"

type admissionLiveRouteWeightedAdmissionResonanceObservationReport struct {
	Schema                                         string `json:"schema"`
	Status                                         string `json:"status"`
	Target                                         string `json:"target"`
	TargetKind                                     string `json:"target_kind"`
	TargetMode                                     string `json:"target_mode"`
	Action                                         string `json:"action"`
	WeightedAdmissionResonanceObservationReady     bool   `json:"weighted_admission_resonance_observation_ready"`
	WeightedAdmissionResonanceReceiverConsumed     bool   `json:"weighted_admission_resonance_receiver_consumed"`
	WeightedAdmissionResonanceReceiverRequired     bool   `json:"weighted_admission_resonance_receiver_required"`
	NextStepBlockedWithoutResonanceObservation     bool   `json:"next_step_blocked_without_resonance_observation"`
	WeightedAdmissionResonanceObservationID        string `json:"weighted_admission_resonance_observation_id"`
	Observer                                       string `json:"observer"`
	ObserverKind                                   string `json:"observer_kind"`
	ObservationKind                                string `json:"observation_kind"`
	ObservationMode                                string `json:"observation_mode"`
	CausalID                                       string `json:"causal_id"`
	AppendHash                                     string `json:"append_hash"`
	ReadBackHash                                   string `json:"read_back_hash"`
	AppendOnly                                     bool   `json:"append_only"`
	ReadBack                                       bool   `json:"read_back"`
	ReceiptVerified                                bool   `json:"receipt_verified"`
	DryRunOnly                                     bool   `json:"dry_run_only"`
	RawDreamTextObserved                           bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                          bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                            bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                            bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                            bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                            bool   `json:"body_mutation_allowed"`
	RollbackRequired                               bool   `json:"rollback_required"`
	SourceSchema                                   string `json:"source_schema"`
	SourceStatus                                   string `json:"source_status"`
	SourceTarget                                   string `json:"source_target"`
	SourceReport                                   string `json:"source_report"`
	SourceResonanceIntentReport                    string `json:"source_resonance_intent_report"`
	SourceFinalGateReport                          string `json:"source_final_gate_report"`
	SourceSealReport                               string `json:"source_seal_report"`
	SourcePermitReport                             string `json:"source_permit_report"`
	SourceAuthorityReport                          string `json:"source_authority_report"`
	SourceContractReport                           string `json:"source_contract_report"`
	SourcePreconditionReport                       string `json:"source_precondition_report"`
	SourceReadinessReport                          string `json:"source_readiness_report"`
	SourceBodyWorkdir                              string `json:"source_body_workdir"`
	SourceBoundaryReport                           string `json:"source_boundary_report"`
	SourceProofLog                                 string `json:"source_proof_log"`
	SourceFinalGateLog                             string `json:"source_final_gate_log"`
	SourceWeightedAdmissionResonanceReceiverID     string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady  bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceReceiverCausal string `json:"source_weighted_admission_resonance_receiver_causal_id"`
	SourceReceiverPreStateHash                     string `json:"source_receiver_pre_state_hash"`
	SourceReceiverPostStateHash                    string `json:"source_receiver_post_state_hash"`
	SourceReceiverStateDeltaHash                   string `json:"source_receiver_state_delta_hash"`
	SourceWeightedAdmissionResonanceIntentConsumed bool   `json:"source_weighted_admission_resonance_intent_consumed"`
	SourceWeightedAdmissionResonanceIntentRequired bool   `json:"source_weighted_admission_resonance_intent_required"`
	SourceWeightedAdmissionResonanceIntentReady    bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateConsumed       bool   `json:"source_weighted_admission_final_gate_consumed"`
	SourceWeightedAdmissionFinalGateRequired       bool   `json:"source_weighted_admission_final_gate_required"`
	SourceWeightedAdmissionFinalGateReady          bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealConsumed            bool   `json:"source_weighted_admission_seal_consumed"`
	SourceWeightedAdmissionSealRequired            bool   `json:"source_weighted_admission_seal_required"`
	SourceWeightedAdmissionSealReady               bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitConsumed          bool   `json:"source_weighted_admission_permit_consumed"`
	SourceWeightedAdmissionPermitRequired          bool   `json:"source_weighted_admission_permit_required"`
	SourceWeightedAdmissionPermitReady             bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed       bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired       bool   `json:"source_weighted_admission_authority_required"`
	SourceManualPermitRequested                    bool   `json:"source_manual_permit_requested"`
	SourcePermitKeyMatched                         bool   `json:"source_permit_key_matched"`
	SourceRawDreamTextAllowed                      bool   `json:"source_raw_dream_text_allowed"`
	SourceRawDreamTextObserved                     bool   `json:"source_raw_dream_text_observed"`
	SourceRawDreamTextForwarded                    bool   `json:"source_raw_dream_text_forwarded"`
	SourceJanusSurfaceAllowed                      bool   `json:"source_janus_surface_allowed"`
	SourceCoocLearningAllowed                      bool   `json:"source_cooc_learning_allowed"`
	SourceDeltaHarvestAllowed                      bool   `json:"source_delta_harvest_allowed"`
	SourceBodyMutationAllowed                      bool   `json:"source_body_mutation_allowed"`
	SourceRollbackRequired                         bool   `json:"source_rollback_required"`
	SourcePreStateHashRequired                     bool   `json:"source_pre_state_hash_required"`
	SourcePostStateHashRequired                    bool   `json:"source_post_state_hash_required"`
	BodySmokeWeighted                              bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                               bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                            bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                   bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                        bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                         bool   `json:"source_authority_granted"`
	AuthorityGranted                               bool   `json:"authority_granted"`
	ContractsReady                                 bool   `json:"contracts_ready"`
	WriteAllowed                                   bool   `json:"write_allowed"`
	AdmissionAllowed                               bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                           bool   `json:"live_admission_enabled"`
	MutatesState                                   bool   `json:"mutates_state"`
	BodyTarget                                     string `json:"body_target"`
	Passed                                         bool   `json:"passed"`
	Reason                                         string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceObservation(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-observation RESONANCE_RECEIVER_REPORT RESONANCE_OBSERVATION_REPORT")
	}
	receiverPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance observation output path missing")
	}
	receiver, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceReceiverReportForAssert(receiverPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceReceiverReportError(receiver, root); err != nil {
		return err
	}
	observation := admissionLiveRouteWeightedAdmissionResonanceObservationReport{
		Schema:     admissionLiveRouteWeightedAdmissionResonanceObservationSchema,
		Status:     "observation_recorded_dry_run",
		Target:     "resonance",
		TargetKind: "weighted_internal_world_observation",
		TargetMode: "append_only_read_back_dry_run",
		Action:     "record_weighted_resonance_receiver_observation_dry_run",
		WeightedAdmissionResonanceObservationReady: true,
		WeightedAdmissionResonanceReceiverConsumed: true,
		WeightedAdmissionResonanceReceiverRequired: true,
		NextStepBlockedWithoutResonanceObservation: true,
		Observer:                    "resonance",
		ObserverKind:                "internal_world",
		ObservationKind:             "weighted_receiver_state_proof",
		ObservationMode:             "sealed_metadata_observation",
		AppendOnly:                  true,
		ReadBack:                    true,
		ReceiptVerified:             true,
		DryRunOnly:                  true,
		RawDreamTextObserved:        false,
		RawDreamTextForwarded:       false,
		JanusSurfaceAllowed:         false,
		CoocLearningAllowed:         false,
		DeltaHarvestAllowed:         false,
		BodyMutationAllowed:         false,
		RollbackRequired:            true,
		SourceSchema:                receiver.Schema,
		SourceStatus:                receiver.Status,
		SourceTarget:                receiver.Target,
		SourceReport:                receiverPath,
		SourceResonanceIntentReport: receiver.SourceReport,
		SourceFinalGateReport:       receiver.SourceFinalGateReport,
		SourceSealReport:            receiver.SourceSealReport,
		SourcePermitReport:          receiver.SourcePermitReport,
		SourceAuthorityReport:       receiver.SourceAuthorityReport,
		SourceContractReport:        receiver.SourceContractReport,
		SourcePreconditionReport:    receiver.SourcePreconditionReport,
		SourceReadinessReport:       receiver.SourceReadinessReport,
		SourceBodyWorkdir:           receiver.SourceBodyWorkdir,
		SourceBoundaryReport:        receiver.SourceBoundaryReport,
		SourceProofLog:              receiver.SourceProofLog,
		SourceFinalGateLog:          receiver.SourceFinalGateLog,
		SourceWeightedAdmissionResonanceReceiverID:     receiver.WeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:  receiver.WeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceReceiverCausal: receiver.CausalID,
		SourceReceiverPreStateHash:                     receiver.PreStateHash,
		SourceReceiverPostStateHash:                    receiver.PostStateHash,
		SourceReceiverStateDeltaHash:                   receiver.StateDeltaHash,
		SourceWeightedAdmissionResonanceIntentConsumed: receiver.WeightedAdmissionResonanceIntentConsumed,
		SourceWeightedAdmissionResonanceIntentRequired: receiver.WeightedAdmissionResonanceIntentRequired,
		SourceWeightedAdmissionResonanceIntentReady:    receiver.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateConsumed:       receiver.SourceWeightedAdmissionFinalGateConsumed,
		SourceWeightedAdmissionFinalGateRequired:       receiver.SourceWeightedAdmissionFinalGateRequired,
		SourceWeightedAdmissionFinalGateReady:          receiver.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealConsumed:            receiver.SourceWeightedAdmissionSealConsumed,
		SourceWeightedAdmissionSealRequired:            receiver.SourceWeightedAdmissionSealRequired,
		SourceWeightedAdmissionSealReady:               receiver.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitConsumed:          receiver.SourceWeightedAdmissionPermitConsumed,
		SourceWeightedAdmissionPermitRequired:          receiver.SourceWeightedAdmissionPermitRequired,
		SourceWeightedAdmissionPermitReady:             receiver.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:       receiver.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:       receiver.SourceWeightedAdmissionAuthorityRequired,
		SourceManualPermitRequested:                    receiver.SourceManualPermitRequested,
		SourcePermitKeyMatched:                         receiver.SourcePermitKeyMatched,
		SourceRawDreamTextAllowed:                      receiver.SourceRawDreamTextAllowed,
		SourceRawDreamTextObserved:                     receiver.RawDreamTextObserved,
		SourceRawDreamTextForwarded:                    receiver.RawDreamTextForwarded,
		SourceJanusSurfaceAllowed:                      receiver.SourceJanusSurfaceAllowed || receiver.JanusSurfaceAllowed,
		SourceCoocLearningAllowed:                      receiver.SourceCoocLearningAllowed || receiver.CoocLearningAllowed,
		SourceDeltaHarvestAllowed:                      receiver.SourceDeltaHarvestAllowed || receiver.DeltaHarvestAllowed,
		SourceBodyMutationAllowed:                      receiver.BodyMutationAllowed,
		SourceRollbackRequired:                         receiver.RollbackRequired && receiver.SourceRollbackRequired,
		SourcePreStateHashRequired:                     receiver.SourcePreStateHashRequired,
		SourcePostStateHashRequired:                    receiver.SourcePostStateHashRequired,
		BodySmokeWeighted:                              receiver.BodySmokeWeighted,
		NanoDirectRunner:                               receiver.NanoDirectRunner,
		NanoDirectFinalGate:                            receiver.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                   receiver.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                        receiver.BoundaryReportFullChain,
		SourceAuthorityGranted:                         receiver.SourceAuthorityGranted,
		AuthorityGranted:                               false,
		ContractsReady:                                 false,
		WriteAllowed:                                   false,
		AdmissionAllowed:                               false,
		LiveAdmissionEnabled:                           false,
		MutatesState:                                   false,
		BodyTarget:                                     "none",
		Passed:                                         true,
		Reason:                                         "weighted resonance observation recorded and read back without body mutation",
	}
	observation.CausalID = admissionLiveRouteWeightedAdmissionResonanceObservationCausalID(observation)
	observation.AppendHash = admissionLiveRouteWeightedAdmissionResonanceObservationAppendHash(observation)
	observation.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceObservationReadBackHash(observation)
	observation.WeightedAdmissionResonanceObservationID = admissionLiveRouteWeightedAdmissionResonanceObservationID(observation)
	if observation.CausalID == "" ||
		observation.AppendHash == "" ||
		observation.ReadBackHash == "" ||
		observation.WeightedAdmissionResonanceObservationID == "" ||
		observation.AppendHash == observation.ReadBackHash {
		return fmt.Errorf("weighted admission resonance observation read-back proof failed")
	}
	raw, err := json.MarshalIndent(observation, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance observation marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance observation write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-observation] pass: resonance_observation_report=%s resonance_receiver_report=%s\n", outputPath, receiverPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceObservationAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-observation-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceObservationReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceObservationReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceObservationReportError(report admissionLiveRouteWeightedAdmissionResonanceObservationReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance observation schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceObservationSchema {
		return fmt.Errorf("weighted admission resonance observation schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceObservationSchema)
	}
	if report.Status != "observation_recorded_dry_run" {
		return fmt.Errorf("weighted admission resonance observation status mismatch: got %q want %q", report.Status, "observation_recorded_dry_run")
	}
	if report.Target != "resonance" {
		return fmt.Errorf("weighted admission resonance observation target mismatch: got %q want %q", report.Target, "resonance")
	}
	if report.TargetKind != "weighted_internal_world_observation" {
		return fmt.Errorf("weighted admission resonance observation target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_observation")
	}
	if report.TargetMode != "append_only_read_back_dry_run" {
		return fmt.Errorf("weighted admission resonance observation target_mode mismatch: got %q want %q", report.TargetMode, "append_only_read_back_dry_run")
	}
	if report.Action != "record_weighted_resonance_receiver_observation_dry_run" {
		return fmt.Errorf("weighted admission resonance observation action mismatch: got %q want %q", report.Action, "record_weighted_resonance_receiver_observation_dry_run")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_observation_ready", report.WeightedAdmissionResonanceObservationReady},
		{"weighted_admission_resonance_receiver_consumed", report.WeightedAdmissionResonanceReceiverConsumed},
		{"weighted_admission_resonance_receiver_required", report.WeightedAdmissionResonanceReceiverRequired},
		{"next_step_blocked_without_resonance_observation", report.NextStepBlockedWithoutResonanceObservation},
		{"append_only", report.AppendOnly},
		{"read_back", report.ReadBack},
		{"receipt_verified", report.ReceiptVerified},
		{"dry_run_only", report.DryRunOnly},
		{"rollback_required", report.RollbackRequired},
		{"source_weighted_admission_resonance_receiver_ready", report.SourceWeightedAdmissionResonanceReceiverReady},
		{"source_weighted_admission_resonance_intent_consumed", report.SourceWeightedAdmissionResonanceIntentConsumed},
		{"source_weighted_admission_resonance_intent_required", report.SourceWeightedAdmissionResonanceIntentRequired},
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
			return fmt.Errorf("weighted admission resonance observation %s not ready", required.name)
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
		{"source_raw_dream_text_observed", report.SourceRawDreamTextObserved},
		{"source_raw_dream_text_forwarded", report.SourceRawDreamTextForwarded},
		{"source_janus_surface_allowed", report.SourceJanusSurfaceAllowed},
		{"source_cooc_learning_allowed", report.SourceCoocLearningAllowed},
		{"source_delta_harvest_allowed", report.SourceDeltaHarvestAllowed},
		{"source_body_mutation_allowed", report.SourceBodyMutationAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance observation opened %s", closed.name)
		}
	}
	for _, pathField := range []struct {
		name  string
		value string
	}{
		{"source_report", report.SourceReport},
		{"source_resonance_intent_report", report.SourceResonanceIntentReport},
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
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
		{"source_weighted_admission_resonance_receiver_causal_id", report.SourceWeightedAdmissionResonanceReceiverCausal},
		{"source_receiver_pre_state_hash", report.SourceReceiverPreStateHash},
		{"source_receiver_post_state_hash", report.SourceReceiverPostStateHash},
		{"source_receiver_state_delta_hash", report.SourceReceiverStateDeltaHash},
	} {
		if strings.TrimSpace(pathField.value) == "" {
			return fmt.Errorf("weighted admission resonance observation %s missing", pathField.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceReceiverSchema {
		return fmt.Errorf("weighted admission resonance observation source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceReceiverSchema)
	}
	if report.SourceStatus != "receiver_previewed_dry_run" {
		return fmt.Errorf("weighted admission resonance observation source_status mismatch: got %q want %q", report.SourceStatus, "receiver_previewed_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance observation source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance observation source receiver id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverCausal, "weighted-resonance-receiver-causal-") {
		return fmt.Errorf("weighted admission resonance observation source receiver causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceReceiverPreStateHash, "weighted-resonance-receiver-pre-") ||
		!strings.HasPrefix(report.SourceReceiverPostStateHash, "weighted-resonance-receiver-post-") ||
		!strings.HasPrefix(report.SourceReceiverStateDeltaHash, "weighted-resonance-receiver-delta-") ||
		report.SourceReceiverPreStateHash == report.SourceReceiverPostStateHash {
		return fmt.Errorf("weighted admission resonance observation source receiver state proof mismatch")
	}
	if report.Observer != "resonance" {
		return fmt.Errorf("weighted admission resonance observation observer mismatch: got %q want %q", report.Observer, "resonance")
	}
	if report.ObserverKind != "internal_world" {
		return fmt.Errorf("weighted admission resonance observation observer_kind mismatch: got %q want %q", report.ObserverKind, "internal_world")
	}
	if report.ObservationKind != "weighted_receiver_state_proof" {
		return fmt.Errorf("weighted admission resonance observation observation_kind mismatch: got %q want %q", report.ObservationKind, "weighted_receiver_state_proof")
	}
	if report.ObservationMode != "sealed_metadata_observation" {
		return fmt.Errorf("weighted admission resonance observation observation_mode mismatch: got %q want %q", report.ObservationMode, "sealed_metadata_observation")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance observation body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceObservationCausalID(report) {
		return fmt.Errorf("weighted admission resonance observation causal_id mismatch")
	}
	if report.AppendHash == "" || report.AppendHash != admissionLiveRouteWeightedAdmissionResonanceObservationAppendHash(report) {
		return fmt.Errorf("weighted admission resonance observation append_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceObservationReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance observation read_back_hash mismatch")
	}
	if report.AppendHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance observation read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceObservationID == "" || report.WeightedAdmissionResonanceObservationID != admissionLiveRouteWeightedAdmissionResonanceObservationID(report) {
		return fmt.Errorf("weighted admission resonance observation id mismatch")
	}
	if report.Reason != "weighted resonance observation recorded and read back without body mutation" {
		return fmt.Errorf("weighted admission resonance observation reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceObservationCausalID(observation admissionLiveRouteWeightedAdmissionResonanceObservationReport) string {
	h := hashJSON(struct {
		SourceReceiverID       string `json:"source_receiver_id"`
		SourceReceiverCausalID string `json:"source_receiver_causal_id"`
		SourceReport           string `json:"source_report"`
		SourceIntentReport     string `json:"source_intent_report"`
		Observer               string `json:"observer"`
		ObserverKind           string `json:"observer_kind"`
		ObservationKind        string `json:"observation_kind"`
	}{
		SourceReceiverID:       observation.SourceWeightedAdmissionResonanceReceiverID,
		SourceReceiverCausalID: observation.SourceWeightedAdmissionResonanceReceiverCausal,
		SourceReport:           observation.SourceReport,
		SourceIntentReport:     observation.SourceResonanceIntentReport,
		Observer:               observation.Observer,
		ObserverKind:           observation.ObserverKind,
		ObservationKind:        observation.ObservationKind,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-observation-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceObservationAppendHash(observation admissionLiveRouteWeightedAdmissionResonanceObservationReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourceReceiverID    string `json:"source_receiver_id"`
		ReceiverPreHash     string `json:"receiver_pre_hash"`
		ReceiverPostHash    string `json:"receiver_post_hash"`
		ReceiverDeltaHash   string `json:"receiver_delta_hash"`
		ObservationMode     string `json:"observation_mode"`
		AppendOnly          bool   `json:"append_only"`
		DryRunOnly          bool   `json:"dry_run_only"`
		RawDreamTextVisible bool   `json:"raw_dream_text_visible"`
	}{
		CausalID:            observation.CausalID,
		SourceReceiverID:    observation.SourceWeightedAdmissionResonanceReceiverID,
		ReceiverPreHash:     observation.SourceReceiverPreStateHash,
		ReceiverPostHash:    observation.SourceReceiverPostStateHash,
		ReceiverDeltaHash:   observation.SourceReceiverStateDeltaHash,
		ObservationMode:     observation.ObservationMode,
		AppendOnly:          observation.AppendOnly,
		DryRunOnly:          observation.DryRunOnly,
		RawDreamTextVisible: observation.RawDreamTextObserved || observation.RawDreamTextForwarded,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-observation-append-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceObservationReadBackHash(observation admissionLiveRouteWeightedAdmissionResonanceObservationReport) string {
	h := hashJSON(struct {
		AppendHash      string `json:"append_hash"`
		SourceReceiver  string `json:"source_receiver_id"`
		ObservationKind string `json:"observation_kind"`
		ReadBack        bool   `json:"read_back"`
		ReceiptVerified bool   `json:"receipt_verified"`
		BodyMutation    bool   `json:"body_mutation"`
	}{
		AppendHash:      observation.AppendHash,
		SourceReceiver:  observation.SourceWeightedAdmissionResonanceReceiverID,
		ObservationKind: observation.ObservationKind,
		ReadBack:        observation.ReadBack,
		ReceiptVerified: observation.ReceiptVerified,
		BodyMutation:    observation.BodyMutationAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-observation-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceObservationID(observation admissionLiveRouteWeightedAdmissionResonanceObservationReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SourceReport            string `json:"source_report"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		AppendHash              string `json:"append_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		AppendOnly              bool   `json:"append_only"`
		ReadBack                bool   `json:"read_back"`
		ReceiptVerified         bool   `json:"receipt_verified"`
		DryRunOnly              bool   `json:"dry_run_only"`
		RawDreamTextObserved    bool   `json:"raw_dream_text_observed"`
		RawDreamTextForwarded   bool   `json:"raw_dream_text_forwarded"`
		JanusSurfaceAllowed     bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed     bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed     bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		RollbackRequired        bool   `json:"rollback_required"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_observation"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityConsumed bool   `json:"source_authority_consumed"`
	}{
		Schema:                  observation.Schema,
		Status:                  observation.Status,
		Action:                  observation.Action,
		SourceReport:            observation.SourceReport,
		SourceReceiverID:        observation.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                observation.CausalID,
		AppendHash:              observation.AppendHash,
		ReadBackHash:            observation.ReadBackHash,
		Ready:                   observation.WeightedAdmissionResonanceObservationReady,
		AppendOnly:              observation.AppendOnly,
		ReadBack:                observation.ReadBack,
		ReceiptVerified:         observation.ReceiptVerified,
		DryRunOnly:              observation.DryRunOnly,
		RawDreamTextObserved:    observation.RawDreamTextObserved,
		RawDreamTextForwarded:   observation.RawDreamTextForwarded,
		JanusSurfaceAllowed:     observation.JanusSurfaceAllowed,
		CoocLearningAllowed:     observation.CoocLearningAllowed,
		DeltaHarvestAllowed:     observation.DeltaHarvestAllowed,
		BodyMutationAllowed:     observation.BodyMutationAllowed,
		RollbackRequired:        observation.RollbackRequired,
		BodyTarget:              observation.BodyTarget,
		WriteAllowed:            observation.WriteAllowed,
		AdmissionAllowed:        observation.AdmissionAllowed,
		LiveAdmissionEnabled:    observation.LiveAdmissionEnabled,
		MutatesState:            observation.MutatesState,
		NextStepBlockedWithout:  observation.NextStepBlockedWithoutResonanceObservation,
		SourceReceiverReady:     observation.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       observation.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    observation.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         observation.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       observation.SourceWeightedAdmissionPermitReady,
		SourceAuthorityConsumed: observation.SourceWeightedAdmissionAuthorityConsumed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-observation-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceObservationReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceObservationReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceObservationReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance observation path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance observation not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance observation not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance observation JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance observation decode failed: %w", err)
	}
	return report, root, nil
}
