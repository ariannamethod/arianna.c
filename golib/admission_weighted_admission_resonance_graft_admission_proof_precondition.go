package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_proof_precondition.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport struct {
	Schema                                                         string `json:"schema"`
	Status                                                         string `json:"status"`
	Target                                                         string `json:"target"`
	TargetKind                                                     string `json:"target_kind"`
	TargetMode                                                     string `json:"target_mode"`
	Action                                                         string `json:"action"`
	WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady bool   `json:"weighted_admission_resonance_graft_admission_proof_precondition_ready"`
	WeightedAdmissionResonanceGraftAdmissionProofConsumed          bool   `json:"weighted_admission_resonance_graft_admission_proof_consumed"`
	WeightedAdmissionResonanceGraftAdmissionProofRequired          bool   `json:"weighted_admission_resonance_graft_admission_proof_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionProof             bool   `json:"next_step_blocked_without_resonance_graft_admission_proof"`
	WeightedAdmissionResonanceGraftAdmissionProofPreconditionID    string `json:"weighted_admission_resonance_graft_admission_proof_precondition_id"`
	ReceiptShape                                                   string `json:"receipt_shape"`
	PreconditionKind                                               string `json:"precondition_kind"`
	PreconditionMode                                               string `json:"precondition_mode"`
	PreconditionStage                                              string `json:"precondition_stage"`
	CausalID                                                       string `json:"causal_id"`
	PreconditionHash                                               string `json:"precondition_hash"`
	ReadBackHash                                                   string `json:"read_back_hash"`
	ProofVerified                                                  bool   `json:"proof_verified"`
	ProofHashVerified                                              bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                          bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                            bool   `json:"store_reader_verified"`
	StoreVerified                                                  bool   `json:"store_verified"`
	CandidateVerified                                              bool   `json:"candidate_verified"`
	GateVerified                                                   bool   `json:"gate_verified"`
	PreflightVerified                                              bool   `json:"preflight_verified"`
	BoundaryVerified                                               bool   `json:"boundary_verified"`
	ObservationVerified                                            bool   `json:"observation_verified"`
	ReceiverVerified                                               bool   `json:"receiver_verified"`
	IntentVerified                                                 bool   `json:"intent_verified"`
	FinalGateVerified                                              bool   `json:"final_gate_verified"`
	SealVerified                                                   bool   `json:"seal_verified"`
	PermitVerified                                                 bool   `json:"permit_verified"`
	AuthorityVerified                                              bool   `json:"authority_verified"`
	AdmissionRequired                                              bool   `json:"admission_required"`
	ShadowOnly                                                     bool   `json:"shadow_only"`
	GraftAllowed                                                   bool   `json:"graft_allowed"`
	DryRunOnly                                                     bool   `json:"dry_run_only"`
	LiveReady                                                      bool   `json:"live_ready"`
	RawDreamTextAllowed                                            bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                           bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                          bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                            bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                            bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                            bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                            bool   `json:"body_mutation_allowed"`
	RollbackRequired                                               bool   `json:"rollback_required"`
	ReadOnly                                                       bool   `json:"read_only"`
	ReplayOnly                                                     bool   `json:"replay_only"`
	SourceSchema                                                   string `json:"source_schema"`
	SourceStatus                                                   string `json:"source_status"`
	SourceTarget                                                   string `json:"source_target"`
	SourceReport                                                   string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofID          string `json:"source_weighted_admission_resonance_graft_admission_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReady       bool   `json:"source_weighted_admission_resonance_graft_admission_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID    string `json:"source_weighted_admission_resonance_graft_admission_proof_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofHash        string `json:"source_weighted_admission_resonance_graft_admission_proof_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack    string `json:"source_weighted_admission_resonance_graft_admission_proof_read_back_hash"`
	SourceProofAction                                              string `json:"source_proof_action"`
	SourceProofReceiptShape                                        string `json:"source_proof_receipt_shape"`
	SourceProofKind                                                string `json:"source_proof_kind"`
	SourceProofMode                                                string `json:"source_proof_mode"`
	SourceProofStage                                               string `json:"source_proof_stage"`
	SourceProofAdmissionRequired                                   bool   `json:"source_proof_admission_required"`
	SourceProofShadowOnly                                          bool   `json:"source_proof_shadow_only"`
	SourceProofGraftAllowed                                        bool   `json:"source_proof_graft_allowed"`
	SourceProofDryRunOnly                                          bool   `json:"source_proof_dry_run_only"`
	SourceProofLiveReady                                           bool   `json:"source_proof_live_ready"`
	SourceProofRawDreamTextAllowed                                 bool   `json:"source_proof_raw_dream_text_allowed"`
	SourceProofRawDreamTextObserved                                bool   `json:"source_proof_raw_dream_text_observed"`
	SourceProofRawDreamTextForwarded                               bool   `json:"source_proof_raw_dream_text_forwarded"`
	SourceProofJanusSurfaceAllowed                                 bool   `json:"source_proof_janus_surface_allowed"`
	SourceProofCoocLearningAllowed                                 bool   `json:"source_proof_cooc_learning_allowed"`
	SourceProofDeltaHarvestAllowed                                 bool   `json:"source_proof_delta_harvest_allowed"`
	SourceProofBodyMutationAllowed                                 bool   `json:"source_proof_body_mutation_allowed"`
	SourceProofRollbackRequired                                    bool   `json:"source_proof_rollback_required"`
	SourceProofReadOnly                                            bool   `json:"source_proof_read_only"`
	SourceProofReplayOnly                                          bool   `json:"source_proof_replay_only"`
	SourceProofAuthorityGranted                                    bool   `json:"source_proof_authority_granted"`
	SourceProofContractsReady                                      bool   `json:"source_proof_contracts_ready"`
	SourceProofWriteAllowed                                        bool   `json:"source_proof_write_allowed"`
	SourceProofAdmissionAllowed                                    bool   `json:"source_proof_admission_allowed"`
	SourceProofLiveAdmissionEnabled                                bool   `json:"source_proof_live_admission_enabled"`
	SourceProofMutatesState                                        bool   `json:"source_proof_mutates_state"`
	SourceProofBodyTarget                                          string `json:"source_proof_body_target"`
	SourceProofPassed                                              bool   `json:"source_proof_passed"`
	SourceProofReason                                              string `json:"source_proof_reason"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID    string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID          string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady       bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateID               string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady            bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftGateID                    string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                 bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightID               string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady            bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady             bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                  string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady               bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                     string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                  bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                    bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                          bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                               bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                             bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                       bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                       bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                              bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                               bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                            bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                   bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                        bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                         bool   `json:"source_authority_granted"`
	AuthorityGranted                                               bool   `json:"authority_granted"`
	ContractsReady                                                 bool   `json:"contracts_ready"`
	WriteAllowed                                                   bool   `json:"write_allowed"`
	AdmissionAllowed                                               bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                           bool   `json:"live_admission_enabled"`
	MutatesState                                                   bool   `json:"mutates_state"`
	BodyTarget                                                     string `json:"body_target"`
	Passed                                                         bool   `json:"passed"`
	Reason                                                         string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPrecondition(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition RESONANCE_GRAFT_ADMISSION_PROOF_REPORT RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT")
	}
	proofPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition output path missing")
	}
	proof, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportForAssert(proofPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofReportError(proof, root); err != nil {
		return err
	}
	precondition := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport{
		Schema:            admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema,
		Status:            "shadow_graft_admission_proof_precondition_satisfied_dry_run",
		Target:            "live_route_admission_next_step",
		TargetKind:        "weighted_internal_world_shadow_graft_admission_proof_precondition",
		TargetMode:        "closed_receipt_precondition_dry_run",
		Action:            "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission",
		ReceiptShape:      "weighted_resonance_shadow_graft_admission_proof_precondition_receipt",
		PreconditionKind:  "shadow_graft_admission_proof_precondition",
		PreconditionMode:  "closed_receipt_consumption",
		PreconditionStage: "pre_live_graft_admission_proof_precondition",
		WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: true,
		WeightedAdmissionResonanceGraftAdmissionProofConsumed:          true,
		WeightedAdmissionResonanceGraftAdmissionProofRequired:          true,
		NextStepBlockedWithoutResonanceGraftAdmissionProof:             true,
		ProofVerified:         true,
		ProofHashVerified:     true,
		ProofReadBackVerified: true,
		StoreReaderVerified:   proof.StoreReaderVerified,
		StoreVerified:         proof.StoreVerified,
		CandidateVerified:     proof.CandidateVerified,
		GateVerified:          proof.GateVerified,
		PreflightVerified:     proof.PreflightVerified,
		BoundaryVerified:      proof.BoundaryVerified,
		ObservationVerified:   proof.ObservationVerified,
		ReceiverVerified:      proof.ReceiverVerified,
		IntentVerified:        proof.IntentVerified,
		FinalGateVerified:     proof.FinalGateVerified,
		SealVerified:          proof.SealVerified,
		PermitVerified:        proof.PermitVerified,
		AuthorityVerified:     proof.AuthorityVerified,
		AdmissionRequired:     true,
		ShadowOnly:            true,
		GraftAllowed:          false,
		DryRunOnly:            true,
		LiveReady:             true,
		RawDreamTextAllowed:   false,
		RawDreamTextObserved:  false,
		RawDreamTextForwarded: false,
		JanusSurfaceAllowed:   false,
		CoocLearningAllowed:   false,
		DeltaHarvestAllowed:   false,
		BodyMutationAllowed:   false,
		RollbackRequired:      true,
		ReadOnly:              true,
		ReplayOnly:            true,
		SourceSchema:          proof.Schema,
		SourceStatus:          proof.Status,
		SourceTarget:          proof.Target,
		SourceReport:          proofPath,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:       proof.WeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:    proof.WeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID: proof.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofHash:     proof.ProofHash,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack: proof.ReadBackHash,
		SourceProofAction:                proof.Action,
		SourceProofReceiptShape:          proof.ReceiptShape,
		SourceProofKind:                  proof.ProofKind,
		SourceProofMode:                  proof.ProofMode,
		SourceProofStage:                 proof.ProofStage,
		SourceProofAdmissionRequired:     proof.AdmissionRequired,
		SourceProofShadowOnly:            proof.ShadowOnly,
		SourceProofGraftAllowed:          proof.GraftAllowed,
		SourceProofDryRunOnly:            proof.DryRunOnly,
		SourceProofLiveReady:             proof.LiveReady,
		SourceProofRawDreamTextAllowed:   proof.RawDreamTextAllowed,
		SourceProofRawDreamTextObserved:  proof.RawDreamTextObserved,
		SourceProofRawDreamTextForwarded: proof.RawDreamTextForwarded,
		SourceProofJanusSurfaceAllowed:   proof.JanusSurfaceAllowed,
		SourceProofCoocLearningAllowed:   proof.CoocLearningAllowed,
		SourceProofDeltaHarvestAllowed:   proof.DeltaHarvestAllowed,
		SourceProofBodyMutationAllowed:   proof.BodyMutationAllowed,
		SourceProofRollbackRequired:      proof.RollbackRequired,
		SourceProofReadOnly:              proof.ReadOnly,
		SourceProofReplayOnly:            proof.ReplayOnly,
		SourceProofAuthorityGranted:      proof.AuthorityGranted,
		SourceProofContractsReady:        proof.ContractsReady,
		SourceProofWriteAllowed:          proof.WriteAllowed,
		SourceProofAdmissionAllowed:      proof.AdmissionAllowed,
		SourceProofLiveAdmissionEnabled:  proof.LiveAdmissionEnabled,
		SourceProofMutatesState:          proof.MutatesState,
		SourceProofBodyTarget:            proof.BodyTarget,
		SourceProofPassed:                proof.Passed,
		SourceProofReason:                proof.Reason,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:    proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady: proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:          proof.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:       proof.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:               proof.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:            proof.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                    proof.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                 proof.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:               proof.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:            proof.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                proof.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:             proof.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                  proof.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:               proof.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                     proof.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                  proof.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                    proof.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                          proof.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                               proof.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                             proof.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                       proof.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                       proof.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                              proof.BodySmokeWeighted,
		NanoDirectRunner:                                               proof.NanoDirectRunner,
		NanoDirectFinalGate:                                            proof.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                   proof.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                        proof.BoundaryReportFullChain,
		SourceAuthorityGranted:                                         proof.SourceAuthorityGranted,
		AuthorityGranted:                                               false,
		ContractsReady:                                                 false,
		WriteAllowed:                                                   false,
		AdmissionAllowed:                                               false,
		LiveAdmissionEnabled:                                           false,
		MutatesState:                                                   false,
		BodyTarget:                                                     "none",
		Passed:                                                         true,
		Reason:                                                         "weighted resonance shadow graft admission proof consumed as closed precondition",
	}
	precondition.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID(precondition)
	precondition.PreconditionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash(precondition)
	precondition.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBackHash(precondition)
	precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionID(precondition)
	if precondition.CausalID == "" ||
		precondition.PreconditionHash == "" ||
		precondition.ReadBackHash == "" ||
		precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID == "" ||
		precondition.PreconditionHash == precondition.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition read-back proof failed")
	}
	raw, err := json.MarshalIndent(precondition, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition] pass: resonance_graft_admission_proof_precondition_report=%s resonance_graft_admission_proof_report=%s\n", outputPath, proofPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-proof-precondition-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema)
	}
	if report.Status != "shadow_graft_admission_proof_precondition_satisfied_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition status mismatch: got %q want %q", report.Status, "shadow_graft_admission_proof_precondition_satisfied_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_proof_precondition" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_proof_precondition")
	}
	if report.TargetMode != "closed_receipt_precondition_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition target_mode mismatch: got %q want %q", report.TargetMode, "closed_receipt_precondition_dry_run")
	}
	if report.Action != "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition action mismatch: got %q want %q", report.Action, "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_proof_precondition_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_proof_precondition_receipt")
	}
	if report.PreconditionKind != "shadow_graft_admission_proof_precondition" ||
		report.PreconditionMode != "closed_receipt_consumption" ||
		report.PreconditionStage != "pre_live_graft_admission_proof_precondition" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_proof_precondition_ready", report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady},
		{"weighted_admission_resonance_graft_admission_proof_consumed", report.WeightedAdmissionResonanceGraftAdmissionProofConsumed},
		{"weighted_admission_resonance_graft_admission_proof_required", report.WeightedAdmissionResonanceGraftAdmissionProofRequired},
		{"next_step_blocked_without_resonance_graft_admission_proof", report.NextStepBlockedWithoutResonanceGraftAdmissionProof},
		{"proof_verified", report.ProofVerified},
		{"proof_hash_verified", report.ProofHashVerified},
		{"proof_read_back_verified", report.ProofReadBackVerified},
		{"store_reader_verified", report.StoreReaderVerified},
		{"store_verified", report.StoreVerified},
		{"candidate_verified", report.CandidateVerified},
		{"gate_verified", report.GateVerified},
		{"preflight_verified", report.PreflightVerified},
		{"boundary_verified", report.BoundaryVerified},
		{"observation_verified", report.ObservationVerified},
		{"receiver_verified", report.ReceiverVerified},
		{"intent_verified", report.IntentVerified},
		{"final_gate_verified", report.FinalGateVerified},
		{"seal_verified", report.SealVerified},
		{"permit_verified", report.PermitVerified},
		{"authority_verified", report.AuthorityVerified},
		{"admission_required", report.AdmissionRequired},
		{"shadow_only", report.ShadowOnly},
		{"dry_run_only", report.DryRunOnly},
		{"live_ready", report.LiveReady},
		{"rollback_required", report.RollbackRequired},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReady},
		{"source_proof_admission_required", report.SourceProofAdmissionRequired},
		{"source_proof_shadow_only", report.SourceProofShadowOnly},
		{"source_proof_dry_run_only", report.SourceProofDryRunOnly},
		{"source_proof_live_ready", report.SourceProofLiveReady},
		{"source_proof_rollback_required", report.SourceProofRollbackRequired},
		{"source_proof_read_only", report.SourceProofReadOnly},
		{"source_proof_replay_only", report.SourceProofReplayOnly},
		{"source_proof_passed", report.SourceProofPassed},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady},
		{"source_weighted_admission_resonance_graft_candidate_store_ready", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReady},
		{"source_weighted_admission_resonance_graft_candidate_ready", report.SourceWeightedAdmissionResonanceGraftCandidateReady},
		{"source_weighted_admission_resonance_graft_gate_ready", report.SourceWeightedAdmissionResonanceGraftGateReady},
		{"source_weighted_admission_resonance_graft_preflight_ready", report.SourceWeightedAdmissionResonanceGraftPreflightReady},
		{"source_weighted_admission_resonance_graft_boundary_ready", report.SourceWeightedAdmissionResonanceGraftBoundaryReady},
		{"source_weighted_admission_resonance_observation_ready", report.SourceWeightedAdmissionResonanceObservationReady},
		{"source_weighted_admission_resonance_receiver_ready", report.SourceWeightedAdmissionResonanceReceiverReady},
		{"source_weighted_admission_resonance_intent_ready", report.SourceWeightedAdmissionResonanceIntentReady},
		{"source_weighted_admission_final_gate_ready", report.SourceWeightedAdmissionFinalGateReady},
		{"source_weighted_admission_seal_ready", report.SourceWeightedAdmissionSealReady},
		{"source_weighted_admission_permit_ready", report.SourceWeightedAdmissionPermitReady},
		{"source_weighted_admission_authority_consumed", report.SourceWeightedAdmissionAuthorityConsumed},
		{"source_weighted_admission_authority_required", report.SourceWeightedAdmissionAuthorityRequired},
		{"body_smoke_weighted", report.BodySmokeWeighted},
		{"nano_direct_runner", report.NanoDirectRunner},
		{"nano_direct_final_gate", report.NanoDirectFinalGate},
		{"resonance_graft_admission_proof", report.ResonanceGraftAdmissionProof},
		{"boundary_report_full_chain", report.BoundaryReportFullChain},
		{"passed", report.Passed},
	} {
		if !required.value {
			return fmt.Errorf("weighted admission resonance graft admission proof precondition %s not ready", required.name)
		}
	}
	for _, closed := range []struct {
		name  string
		value bool
	}{
		{"graft_allowed", report.GraftAllowed},
		{"raw_dream_text_allowed", report.RawDreamTextAllowed},
		{"raw_dream_text_observed", report.RawDreamTextObserved},
		{"raw_dream_text_forwarded", report.RawDreamTextForwarded},
		{"janus_surface_allowed", report.JanusSurfaceAllowed},
		{"cooc_learning_allowed", report.CoocLearningAllowed},
		{"delta_harvest_allowed", report.DeltaHarvestAllowed},
		{"body_mutation_allowed", report.BodyMutationAllowed},
		{"source_proof_graft_allowed", report.SourceProofGraftAllowed},
		{"source_proof_raw_dream_text_allowed", report.SourceProofRawDreamTextAllowed},
		{"source_proof_raw_dream_text_observed", report.SourceProofRawDreamTextObserved},
		{"source_proof_raw_dream_text_forwarded", report.SourceProofRawDreamTextForwarded},
		{"source_proof_janus_surface_allowed", report.SourceProofJanusSurfaceAllowed},
		{"source_proof_cooc_learning_allowed", report.SourceProofCoocLearningAllowed},
		{"source_proof_delta_harvest_allowed", report.SourceProofDeltaHarvestAllowed},
		{"source_proof_body_mutation_allowed", report.SourceProofBodyMutationAllowed},
		{"source_proof_authority_granted", report.SourceProofAuthorityGranted},
		{"source_proof_contracts_ready", report.SourceProofContractsReady},
		{"source_proof_write_allowed", report.SourceProofWriteAllowed},
		{"source_proof_admission_allowed", report.SourceProofAdmissionAllowed},
		{"source_proof_live_admission_enabled", report.SourceProofLiveAdmissionEnabled},
		{"source_proof_mutates_state", report.SourceProofMutatesState},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission proof precondition opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_proof_precondition_id", report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID},
		{"causal_id", report.CausalID},
		{"precondition_hash", report.PreconditionHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofID},
		{"source_weighted_admission_resonance_graft_admission_proof_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID},
		{"source_weighted_admission_resonance_graft_admission_proof_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionProofHash},
		{"source_weighted_admission_resonance_graft_admission_proof_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack},
		{"source_weighted_admission_resonance_graft_candidate_store_reader_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID},
		{"source_weighted_admission_resonance_graft_candidate_store_id", report.SourceWeightedAdmissionResonanceGraftCandidateStoreID},
		{"source_weighted_admission_resonance_graft_candidate_id", report.SourceWeightedAdmissionResonanceGraftCandidateID},
		{"source_weighted_admission_resonance_graft_gate_id", report.SourceWeightedAdmissionResonanceGraftGateID},
		{"source_weighted_admission_resonance_graft_preflight_id", report.SourceWeightedAdmissionResonanceGraftPreflightID},
		{"source_weighted_admission_resonance_graft_boundary_id", report.SourceWeightedAdmissionResonanceGraftBoundaryID},
		{"source_weighted_admission_resonance_observation_id", report.SourceWeightedAdmissionResonanceObservationID},
		{"source_weighted_admission_resonance_receiver_id", report.SourceWeightedAdmissionResonanceReceiverID},
	} {
		if strings.TrimSpace(requiredString.value) == "" {
			return fmt.Errorf("weighted admission resonance graft admission proof precondition %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_proof_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_proof_ready_dry_run")
	}
	if report.SourceTarget != "resonance" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_target mismatch: got %q want %q", report.SourceTarget, "resonance")
	}
	if report.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_proof_action mismatch: got %q want %q", report.SourceProofAction, "prove_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_proof_receipt_shape mismatch: got %q want %q", report.SourceProofReceiptShape, "weighted_resonance_shadow_graft_admission_proof_receipt")
	}
	if report.SourceProofKind != "shadow_graft_admission_proof" ||
		report.SourceProofMode != "closed_read_back_admission_proof" ||
		report.SourceProofStage != "pre_live_graft_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source proof shape mismatch")
	}
	if report.SourceProofBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source_proof_body_target mismatch: got %q want %q", report.SourceProofBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-proof-precondition-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition causal prefix mismatch")
	}
	if !strings.HasPrefix(report.PreconditionHash, "weighted-resonance-graft-admission-proof-precondition-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-proof-precondition-read-") ||
		report.PreconditionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID, "weighted-resonance-graft-admission-proof-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofHash, "weighted-resonance-graft-admission-proof-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack, "weighted-resonance-graft-admission-proof-read-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition causal_id mismatch")
	}
	if report.PreconditionHash == "" || report.PreconditionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition precondition_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition read_back_hash mismatch")
	}
	if report.PreconditionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID == "" || report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission proof consumed as closed precondition" {
		return fmt.Errorf("weighted admission resonance graft admission proof precondition reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport) string {
	h := hashJSON(struct {
		SourceProofID     string `json:"source_proof_id"`
		SourceProofRead   string `json:"source_proof_read_back_hash"`
		SourceReaderID    string `json:"source_reader_id"`
		SourceStoreID     string `json:"source_store_id"`
		SourceCandidateID string `json:"source_candidate_id"`
		SourceGateID      string `json:"source_gate_id"`
		SourceObservation string `json:"source_observation_id"`
		Target            string `json:"target"`
		PreconditionKind  string `json:"precondition_kind"`
		PreconditionStage string `json:"precondition_stage"`
	}{
		SourceProofID:     precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceProofRead:   precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack,
		SourceReaderID:    precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:     precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID: precondition.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:      precondition.SourceWeightedAdmissionResonanceGraftGateID,
		SourceObservation: precondition.SourceWeightedAdmissionResonanceObservationID,
		Target:            precondition.Target,
		PreconditionKind:  precondition.PreconditionKind,
		PreconditionStage: precondition.PreconditionStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-precondition-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport) string {
	h := hashJSON(struct {
		CausalID             string `json:"causal_id"`
		SourceProofID        string `json:"source_proof_id"`
		SourceProofHash      string `json:"source_proof_hash"`
		SourceProofRead      string `json:"source_proof_read_back_hash"`
		Action               string `json:"action"`
		ReceiptShape         string `json:"receipt_shape"`
		PreconditionMode     string `json:"precondition_mode"`
		ProofVerified        bool   `json:"proof_verified"`
		ProofHashVerified    bool   `json:"proof_hash_verified"`
		ProofReadVerified    bool   `json:"proof_read_back_verified"`
		ReadOnly             bool   `json:"read_only"`
		ReplayOnly           bool   `json:"replay_only"`
		AdmissionRequired    bool   `json:"admission_required"`
		ShadowOnly           bool   `json:"shadow_only"`
		DryRunOnly           bool   `json:"dry_run_only"`
		GraftAllowed         bool   `json:"graft_allowed"`
		BodyMutationAllowed  bool   `json:"body_mutation_allowed"`
		LiveAdmissionEnabled bool   `json:"live_admission_enabled"`
	}{
		CausalID:             precondition.CausalID,
		SourceProofID:        precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceProofHash:      precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofHash,
		SourceProofRead:      precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack,
		Action:               precondition.Action,
		ReceiptShape:         precondition.ReceiptShape,
		PreconditionMode:     precondition.PreconditionMode,
		ProofVerified:        precondition.ProofVerified,
		ProofHashVerified:    precondition.ProofHashVerified,
		ProofReadVerified:    precondition.ProofReadBackVerified,
		ReadOnly:             precondition.ReadOnly,
		ReplayOnly:           precondition.ReplayOnly,
		AdmissionRequired:    precondition.AdmissionRequired,
		ShadowOnly:           precondition.ShadowOnly,
		DryRunOnly:           precondition.DryRunOnly,
		GraftAllowed:         precondition.GraftAllowed,
		BodyMutationAllowed:  precondition.BodyMutationAllowed,
		LiveAdmissionEnabled: precondition.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-precondition-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBackHash(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport) string {
	h := hashJSON(struct {
		PreconditionHash  string `json:"precondition_hash"`
		SourceProofID     string `json:"source_proof_id"`
		SourceProofRead   string `json:"source_proof_read_back_hash"`
		PreconditionKind  string `json:"precondition_kind"`
		PreconditionReady bool   `json:"precondition_ready"`
		BodyMutation      bool   `json:"body_mutation"`
		LiveAdmission     bool   `json:"live_admission"`
		WriteAllowed      bool   `json:"write_allowed"`
		AdmissionAllowed  bool   `json:"admission_allowed"`
	}{
		PreconditionHash:  precondition.PreconditionHash,
		SourceProofID:     precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceProofRead:   precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack,
		PreconditionKind:  precondition.PreconditionKind,
		PreconditionReady: precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		BodyMutation:      precondition.BodyMutationAllowed,
		LiveAdmission:     precondition.LiveAdmissionEnabled,
		WriteAllowed:      precondition.WriteAllowed,
		AdmissionAllowed:  precondition.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-precondition-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionID(precondition admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport) string {
	h := hashJSON(struct {
		Schema                 string `json:"schema"`
		Status                 string `json:"status"`
		Action                 string `json:"action"`
		SourceReport           string `json:"source_report"`
		SourceProofID          string `json:"source_proof_id"`
		SourceReaderID         string `json:"source_reader_id"`
		SourceStoreID          string `json:"source_store_id"`
		SourceCandidateID      string `json:"source_candidate_id"`
		SourceGateID           string `json:"source_gate_id"`
		SourcePreflightID      string `json:"source_preflight_id"`
		SourceBoundaryID       string `json:"source_boundary_id"`
		SourceObservationID    string `json:"source_observation_id"`
		SourceReceiverID       string `json:"source_receiver_id"`
		CausalID               string `json:"causal_id"`
		PreconditionHash       string `json:"precondition_hash"`
		ReadBackHash           string `json:"read_back_hash"`
		Ready                  bool   `json:"ready"`
		ReceiptShape           string `json:"receipt_shape"`
		PreconditionKind       string `json:"precondition_kind"`
		PreconditionMode       string `json:"precondition_mode"`
		PreconditionStage      string `json:"precondition_stage"`
		ProofVerified          bool   `json:"proof_verified"`
		ProofHashVerified      bool   `json:"proof_hash_verified"`
		ProofReadBackVerified  bool   `json:"proof_read_back_verified"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
		DryRunOnly             bool   `json:"dry_run_only"`
		RawDreamTextAllowed    bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed    bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed    bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed    bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		RollbackRequired       bool   `json:"rollback_required"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		LiveReady              bool   `json:"live_ready"`
		ContractsReady         bool   `json:"contracts_ready"`
		BodyTarget             string `json:"body_target"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
		MutatesState           bool   `json:"mutates_state"`
		NextStepBlockedWithout bool   `json:"next_step_blocked_without_resonance_graft_admission_proof"`
		SourceProofReady       bool   `json:"source_proof_ready"`
		SourceReaderReady      bool   `json:"source_reader_ready"`
		SourceStoreReady       bool   `json:"source_store_ready"`
		SourceCandidateReady   bool   `json:"source_candidate_ready"`
		SourceGateReady        bool   `json:"source_gate_ready"`
		SourcePreflightReady   bool   `json:"source_preflight_ready"`
		SourceBoundaryReady    bool   `json:"source_boundary_ready"`
		SourceObservationReady bool   `json:"source_observation_ready"`
		SourceReceiverReady    bool   `json:"source_receiver_ready"`
		SourceIntentReady      bool   `json:"source_intent_ready"`
		SourceFinalGateReady   bool   `json:"source_final_gate_ready"`
		SourceSealReady        bool   `json:"source_seal_ready"`
		SourcePermitReady      bool   `json:"source_permit_ready"`
		SourceAuthorityUsed    bool   `json:"source_authority_consumed"`
		SourceAuthorityNeeded  bool   `json:"source_authority_required"`
	}{
		Schema:                 precondition.Schema,
		Status:                 precondition.Status,
		Action:                 precondition.Action,
		SourceReport:           precondition.SourceReport,
		SourceProofID:          precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:         precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:          precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:      precondition.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:           precondition.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:      precondition.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:       precondition.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:    precondition.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:       precondition.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:               precondition.CausalID,
		PreconditionHash:       precondition.PreconditionHash,
		ReadBackHash:           precondition.ReadBackHash,
		Ready:                  precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		ReceiptShape:           precondition.ReceiptShape,
		PreconditionKind:       precondition.PreconditionKind,
		PreconditionMode:       precondition.PreconditionMode,
		PreconditionStage:      precondition.PreconditionStage,
		ProofVerified:          precondition.ProofVerified,
		ProofHashVerified:      precondition.ProofHashVerified,
		ProofReadBackVerified:  precondition.ProofReadBackVerified,
		AdmissionRequired:      precondition.AdmissionRequired,
		ShadowOnly:             precondition.ShadowOnly,
		GraftAllowed:           precondition.GraftAllowed,
		DryRunOnly:             precondition.DryRunOnly,
		RawDreamTextAllowed:    precondition.RawDreamTextAllowed,
		JanusSurfaceAllowed:    precondition.JanusSurfaceAllowed,
		CoocLearningAllowed:    precondition.CoocLearningAllowed,
		DeltaHarvestAllowed:    precondition.DeltaHarvestAllowed,
		BodyMutationAllowed:    precondition.BodyMutationAllowed,
		RollbackRequired:       precondition.RollbackRequired,
		ReadOnly:               precondition.ReadOnly,
		ReplayOnly:             precondition.ReplayOnly,
		LiveReady:              precondition.LiveReady,
		ContractsReady:         precondition.ContractsReady,
		BodyTarget:             precondition.BodyTarget,
		WriteAllowed:           precondition.WriteAllowed,
		AdmissionAllowed:       precondition.AdmissionAllowed,
		LiveAdmissionEnabled:   precondition.LiveAdmissionEnabled,
		MutatesState:           precondition.MutatesState,
		NextStepBlockedWithout: precondition.NextStepBlockedWithoutResonanceGraftAdmissionProof,
		SourceProofReady:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:      precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:       precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:   precondition.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:        precondition.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:   precondition.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:    precondition.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady: precondition.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:    precondition.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:      precondition.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:   precondition.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:        precondition.SourceWeightedAdmissionSealReady,
		SourcePermitReady:      precondition.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:    precondition.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:  precondition.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-proof-precondition-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof precondition path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission proof precondition not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof precondition not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof precondition JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission proof precondition decode failed: %w", err)
	}
	return report, root, nil
}
