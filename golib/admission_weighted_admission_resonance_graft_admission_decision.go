package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_decision.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport struct {
	Schema                                                                  string `json:"schema"`
	Status                                                                  string `json:"status"`
	Target                                                                  string `json:"target"`
	TargetKind                                                              string `json:"target_kind"`
	TargetMode                                                              string `json:"target_mode"`
	Action                                                                  string `json:"action"`
	Decision                                                                string `json:"decision"`
	WeightedAdmissionResonanceGraftAdmissionDecisionReady                   bool   `json:"weighted_admission_resonance_graft_admission_decision_ready"`
	WeightedAdmissionResonanceGraftAdmissionProofPreconditionConsumed       bool   `json:"weighted_admission_resonance_graft_admission_proof_precondition_consumed"`
	WeightedAdmissionResonanceGraftAdmissionProofPreconditionRequired       bool   `json:"weighted_admission_resonance_graft_admission_proof_precondition_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionDecision                   bool   `json:"next_step_blocked_without_resonance_graft_admission_decision"`
	WeightedAdmissionResonanceGraftAdmissionDecisionID                      string `json:"weighted_admission_resonance_graft_admission_decision_id"`
	ReceiptShape                                                            string `json:"receipt_shape"`
	DecisionKind                                                            string `json:"decision_kind"`
	DecisionMode                                                            string `json:"decision_mode"`
	DecisionStage                                                           string `json:"decision_stage"`
	CausalID                                                                string `json:"causal_id"`
	DecisionHash                                                            string `json:"decision_hash"`
	ReadBackHash                                                            string `json:"read_back_hash"`
	ProofPreconditionVerified                                               bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                                bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                            bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                           bool   `json:"proof_verified"`
	ProofHashVerified                                                       bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                   bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                     bool   `json:"store_reader_verified"`
	StoreVerified                                                           bool   `json:"store_verified"`
	CandidateVerified                                                       bool   `json:"candidate_verified"`
	GateVerified                                                            bool   `json:"gate_verified"`
	PreflightVerified                                                       bool   `json:"preflight_verified"`
	BoundaryVerified                                                        bool   `json:"boundary_verified"`
	ObservationVerified                                                     bool   `json:"observation_verified"`
	ReceiverVerified                                                        bool   `json:"receiver_verified"`
	IntentVerified                                                          bool   `json:"intent_verified"`
	FinalGateVerified                                                       bool   `json:"final_gate_verified"`
	SealVerified                                                            bool   `json:"seal_verified"`
	PermitVerified                                                          bool   `json:"permit_verified"`
	AuthorityVerified                                                       bool   `json:"authority_verified"`
	AdmissionRequired                                                       bool   `json:"admission_required"`
	ShadowOnly                                                              bool   `json:"shadow_only"`
	GraftAllowed                                                            bool   `json:"graft_allowed"`
	DryRunOnly                                                              bool   `json:"dry_run_only"`
	LiveReady                                                               bool   `json:"live_ready"`
	RawDreamTextAllowed                                                     bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                    bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                   bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                     bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                     bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                     bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                     bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                        bool   `json:"rollback_required"`
	ReadOnly                                                                bool   `json:"read_only"`
	ReplayOnly                                                              bool   `json:"replay_only"`
	SourceSchema                                                            string `json:"source_schema"`
	SourceStatus                                                            string `json:"source_status"`
	SourceTarget                                                            string `json:"source_target"`
	SourceReport                                                            string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID       string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady    bool   `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash     string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_read_back_hash"`
	SourcePreconditionAction                                                string `json:"source_precondition_action"`
	SourcePreconditionReceiptShape                                          string `json:"source_precondition_receipt_shape"`
	SourcePreconditionKind                                                  string `json:"source_precondition_kind"`
	SourcePreconditionMode                                                  string `json:"source_precondition_mode"`
	SourcePreconditionStage                                                 string `json:"source_precondition_stage"`
	SourcePreconditionAdmissionRequired                                     bool   `json:"source_precondition_admission_required"`
	SourcePreconditionShadowOnly                                            bool   `json:"source_precondition_shadow_only"`
	SourcePreconditionGraftAllowed                                          bool   `json:"source_precondition_graft_allowed"`
	SourcePreconditionDryRunOnly                                            bool   `json:"source_precondition_dry_run_only"`
	SourcePreconditionLiveReady                                             bool   `json:"source_precondition_live_ready"`
	SourcePreconditionRawDreamTextAllowed                                   bool   `json:"source_precondition_raw_dream_text_allowed"`
	SourcePreconditionRawDreamTextObserved                                  bool   `json:"source_precondition_raw_dream_text_observed"`
	SourcePreconditionRawDreamTextForwarded                                 bool   `json:"source_precondition_raw_dream_text_forwarded"`
	SourcePreconditionJanusSurfaceAllowed                                   bool   `json:"source_precondition_janus_surface_allowed"`
	SourcePreconditionCoocLearningAllowed                                   bool   `json:"source_precondition_cooc_learning_allowed"`
	SourcePreconditionDeltaHarvestAllowed                                   bool   `json:"source_precondition_delta_harvest_allowed"`
	SourcePreconditionBodyMutationAllowed                                   bool   `json:"source_precondition_body_mutation_allowed"`
	SourcePreconditionRollbackRequired                                      bool   `json:"source_precondition_rollback_required"`
	SourcePreconditionReadOnly                                              bool   `json:"source_precondition_read_only"`
	SourcePreconditionReplayOnly                                            bool   `json:"source_precondition_replay_only"`
	SourcePreconditionBodyTarget                                            string `json:"source_precondition_body_target"`
	SourcePreconditionPassed                                                bool   `json:"source_precondition_passed"`
	SourcePreconditionReason                                                string `json:"source_precondition_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofID                   string `json:"source_weighted_admission_resonance_graft_admission_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReady                bool   `json:"source_weighted_admission_resonance_graft_admission_proof_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID             string `json:"source_weighted_admission_resonance_graft_admission_proof_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofHash                 string `json:"source_weighted_admission_resonance_graft_admission_proof_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack             string `json:"source_weighted_admission_resonance_graft_admission_proof_read_back_hash"`
	SourceProofAction                                                       string `json:"source_proof_action"`
	SourceProofReceiptShape                                                 string `json:"source_proof_receipt_shape"`
	SourceProofKind                                                         string `json:"source_proof_kind"`
	SourceProofMode                                                         string `json:"source_proof_mode"`
	SourceProofStage                                                        string `json:"source_proof_stage"`
	SourceProofGraftAllowed                                                 bool   `json:"source_proof_graft_allowed"`
	SourceProofLiveAdmissionEnabled                                         bool   `json:"source_proof_live_admission_enabled"`
	SourceProofBodyMutationAllowed                                          bool   `json:"source_proof_body_mutation_allowed"`
	SourceProofBodyTarget                                                   string `json:"source_proof_body_target"`
	SourceProofPassed                                                       bool   `json:"source_proof_passed"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID             string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady          bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID                   string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady                bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateID                        string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady                     bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftGateID                             string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                          bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightID                        string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady                     bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                         string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady                      bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                           string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                        bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                              string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                           bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                             bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                                   bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                        bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                                      bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                                bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                                bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                                       bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                        bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                     bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                            bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                                 bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                                  bool   `json:"source_authority_granted"`
	AuthorityGranted                                                        bool   `json:"authority_granted"`
	ContractsReady                                                          bool   `json:"contracts_ready"`
	WriteAllowed                                                            bool   `json:"write_allowed"`
	AdmissionAllowed                                                        bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                    bool   `json:"live_admission_enabled"`
	MutatesState                                                            bool   `json:"mutates_state"`
	BodyTarget                                                              string `json:"body_target"`
	Passed                                                                  bool   `json:"passed"`
	Reason                                                                  string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecision(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision RESONANCE_GRAFT_ADMISSION_PROOF_PRECONDITION_REPORT RESONANCE_GRAFT_ADMISSION_DECISION_REPORT")
	}
	preconditionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission decision output path missing")
	}
	precondition, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportForAssert(preconditionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionReportError(precondition, root); err != nil {
		return err
	}
	decision := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport{
		Schema:        admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema,
		Status:        "shadow_graft_admission_decision_ready_dry_run",
		Target:        "live_route_admission_next_step",
		TargetKind:    "weighted_internal_world_shadow_graft_admission_decision",
		TargetMode:    "closed_decision_receipt_dry_run",
		Action:        "decide_weighted_resonance_shadow_graft_admission_dry_run",
		Decision:      "shadow_ready",
		ReceiptShape:  "weighted_resonance_shadow_graft_admission_decision_receipt",
		DecisionKind:  "shadow_graft_admission_decision",
		DecisionMode:  "closed_precondition_decision",
		DecisionStage: "pre_live_graft_admission_decision",
		WeightedAdmissionResonanceGraftAdmissionDecisionReady:             true,
		WeightedAdmissionResonanceGraftAdmissionProofPreconditionConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionProofPreconditionRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionDecision:             true,
		ProofPreconditionVerified:                                         true,
		PreconditionHashVerified:                                          true,
		PreconditionReadBackVerified:                                      true,
		ProofVerified:                                                     precondition.ProofVerified,
		ProofHashVerified:                                                 precondition.ProofHashVerified,
		ProofReadBackVerified:                                             precondition.ProofReadBackVerified,
		StoreReaderVerified:                                               precondition.StoreReaderVerified,
		StoreVerified:                                                     precondition.StoreVerified,
		CandidateVerified:                                                 precondition.CandidateVerified,
		GateVerified:                                                      precondition.GateVerified,
		PreflightVerified:                                                 precondition.PreflightVerified,
		BoundaryVerified:                                                  precondition.BoundaryVerified,
		ObservationVerified:                                               precondition.ObservationVerified,
		ReceiverVerified:                                                  precondition.ReceiverVerified,
		IntentVerified:                                                    precondition.IntentVerified,
		FinalGateVerified:                                                 precondition.FinalGateVerified,
		SealVerified:                                                      precondition.SealVerified,
		PermitVerified:                                                    precondition.PermitVerified,
		AuthorityVerified:                                                 precondition.AuthorityVerified,
		AdmissionRequired:                                                 true,
		ShadowOnly:                                                        true,
		GraftAllowed:                                                      false,
		DryRunOnly:                                                        true,
		LiveReady:                                                         true,
		RawDreamTextAllowed:                                               false,
		RawDreamTextObserved:                                              false,
		RawDreamTextForwarded:                                             false,
		JanusSurfaceAllowed:                                               false,
		CoocLearningAllowed:                                               false,
		DeltaHarvestAllowed:                                               false,
		BodyMutationAllowed:                                               false,
		RollbackRequired:                                                  true,
		ReadOnly:                                                          true,
		ReplayOnly:                                                        true,
		SourceSchema:                                                      precondition.Schema,
		SourceStatus:                                                      precondition.Status,
		SourceTarget:                                                      precondition.Target,
		SourceReport:                                                      preconditionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:       precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady:    precondition.WeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID: precondition.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash:     precondition.PreconditionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack: precondition.ReadBackHash,
		SourcePreconditionAction:                                       precondition.Action,
		SourcePreconditionReceiptShape:                                 precondition.ReceiptShape,
		SourcePreconditionKind:                                         precondition.PreconditionKind,
		SourcePreconditionMode:                                         precondition.PreconditionMode,
		SourcePreconditionStage:                                        precondition.PreconditionStage,
		SourcePreconditionAdmissionRequired:                            precondition.AdmissionRequired,
		SourcePreconditionShadowOnly:                                   precondition.ShadowOnly,
		SourcePreconditionGraftAllowed:                                 precondition.GraftAllowed,
		SourcePreconditionDryRunOnly:                                   precondition.DryRunOnly,
		SourcePreconditionLiveReady:                                    precondition.LiveReady,
		SourcePreconditionRawDreamTextAllowed:                          precondition.RawDreamTextAllowed,
		SourcePreconditionRawDreamTextObserved:                         precondition.RawDreamTextObserved,
		SourcePreconditionRawDreamTextForwarded:                        precondition.RawDreamTextForwarded,
		SourcePreconditionJanusSurfaceAllowed:                          precondition.JanusSurfaceAllowed,
		SourcePreconditionCoocLearningAllowed:                          precondition.CoocLearningAllowed,
		SourcePreconditionDeltaHarvestAllowed:                          precondition.DeltaHarvestAllowed,
		SourcePreconditionBodyMutationAllowed:                          precondition.BodyMutationAllowed,
		SourcePreconditionRollbackRequired:                             precondition.RollbackRequired,
		SourcePreconditionReadOnly:                                     precondition.ReadOnly,
		SourcePreconditionReplayOnly:                                   precondition.ReplayOnly,
		SourcePreconditionBodyTarget:                                   precondition.BodyTarget,
		SourcePreconditionPassed:                                       precondition.Passed,
		SourcePreconditionReason:                                       precondition.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:          precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:       precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID:    precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofHash:        precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofHash,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack:    precondition.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack,
		SourceProofAction:                                              precondition.SourceProofAction,
		SourceProofReceiptShape:                                        precondition.SourceProofReceiptShape,
		SourceProofKind:                                                precondition.SourceProofKind,
		SourceProofMode:                                                precondition.SourceProofMode,
		SourceProofStage:                                               precondition.SourceProofStage,
		SourceProofGraftAllowed:                                        precondition.SourceProofGraftAllowed,
		SourceProofLiveAdmissionEnabled:                                precondition.SourceProofLiveAdmissionEnabled,
		SourceProofBodyMutationAllowed:                                 precondition.SourceProofBodyMutationAllowed,
		SourceProofBodyTarget:                                          precondition.SourceProofBodyTarget,
		SourceProofPassed:                                              precondition.SourceProofPassed,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:    precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady: precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:          precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:       precondition.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:               precondition.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:            precondition.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                    precondition.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                 precondition.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:               precondition.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:            precondition.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                precondition.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:             precondition.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                  precondition.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:               precondition.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                     precondition.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                  precondition.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                    precondition.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                          precondition.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                               precondition.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                             precondition.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                       precondition.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                       precondition.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                              precondition.BodySmokeWeighted,
		NanoDirectRunner:                                               precondition.NanoDirectRunner,
		NanoDirectFinalGate:                                            precondition.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                   precondition.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                        precondition.BoundaryReportFullChain,
		SourceAuthorityGranted:                                         precondition.SourceAuthorityGranted,
		AuthorityGranted:                                               false,
		ContractsReady:                                                 false,
		WriteAllowed:                                                   false,
		AdmissionAllowed:                                               false,
		LiveAdmissionEnabled:                                           false,
		MutatesState:                                                   false,
		BodyTarget:                                                     "none",
		Passed:                                                         true,
		Reason:                                                         "weighted resonance shadow graft admission decision accepted precondition as closed shadow-ready receipt",
	}
	decision.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionCausalID(decision)
	decision.DecisionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionHash(decision)
	decision.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReadBackHash(decision)
	decision.WeightedAdmissionResonanceGraftAdmissionDecisionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionID(decision)
	if decision.CausalID == "" ||
		decision.DecisionHash == "" ||
		decision.ReadBackHash == "" ||
		decision.WeightedAdmissionResonanceGraftAdmissionDecisionID == "" ||
		decision.DecisionHash == decision.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission decision read-back proof failed")
	}
	raw, err := json.MarshalIndent(decision, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission decision marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission decision write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-decision] pass: resonance_graft_admission_decision_report=%s resonance_graft_admission_proof_precondition_report=%s\n", outputPath, preconditionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-decision-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission decision schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema {
		return fmt.Errorf("weighted admission resonance graft admission decision schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema)
	}
	if report.Status != "shadow_graft_admission_decision_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission decision status mismatch: got %q want %q", report.Status, "shadow_graft_admission_decision_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission decision target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission decision target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_decision")
	}
	if report.TargetMode != "closed_decision_receipt_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission decision target_mode mismatch: got %q want %q", report.TargetMode, "closed_decision_receipt_dry_run")
	}
	if report.Action != "decide_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission decision action mismatch: got %q want %q", report.Action, "decide_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.Decision != "shadow_ready" {
		return fmt.Errorf("weighted admission resonance graft admission decision decision mismatch: got %q want %q", report.Decision, "shadow_ready")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_decision_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission decision receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_decision_receipt")
	}
	if report.DecisionKind != "shadow_graft_admission_decision" ||
		report.DecisionMode != "closed_precondition_decision" ||
		report.DecisionStage != "pre_live_graft_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission decision shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_decision_ready", report.WeightedAdmissionResonanceGraftAdmissionDecisionReady},
		{"weighted_admission_resonance_graft_admission_proof_precondition_consumed", report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionConsumed},
		{"weighted_admission_resonance_graft_admission_proof_precondition_required", report.WeightedAdmissionResonanceGraftAdmissionProofPreconditionRequired},
		{"next_step_blocked_without_resonance_graft_admission_decision", report.NextStepBlockedWithoutResonanceGraftAdmissionDecision},
		{"proof_precondition_verified", report.ProofPreconditionVerified},
		{"precondition_hash_verified", report.PreconditionHashVerified},
		{"precondition_read_back_verified", report.PreconditionReadBackVerified},
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
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady},
		{"source_precondition_admission_required", report.SourcePreconditionAdmissionRequired},
		{"source_precondition_shadow_only", report.SourcePreconditionShadowOnly},
		{"source_precondition_dry_run_only", report.SourcePreconditionDryRunOnly},
		{"source_precondition_live_ready", report.SourcePreconditionLiveReady},
		{"source_precondition_rollback_required", report.SourcePreconditionRollbackRequired},
		{"source_precondition_read_only", report.SourcePreconditionReadOnly},
		{"source_precondition_replay_only", report.SourcePreconditionReplayOnly},
		{"source_precondition_passed", report.SourcePreconditionPassed},
		{"source_weighted_admission_resonance_graft_admission_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReady},
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
			return fmt.Errorf("weighted admission resonance graft admission decision %s not ready", required.name)
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
		{"source_precondition_graft_allowed", report.SourcePreconditionGraftAllowed},
		{"source_precondition_raw_dream_text_allowed", report.SourcePreconditionRawDreamTextAllowed},
		{"source_precondition_raw_dream_text_observed", report.SourcePreconditionRawDreamTextObserved},
		{"source_precondition_raw_dream_text_forwarded", report.SourcePreconditionRawDreamTextForwarded},
		{"source_precondition_janus_surface_allowed", report.SourcePreconditionJanusSurfaceAllowed},
		{"source_precondition_cooc_learning_allowed", report.SourcePreconditionCoocLearningAllowed},
		{"source_precondition_delta_harvest_allowed", report.SourcePreconditionDeltaHarvestAllowed},
		{"source_precondition_body_mutation_allowed", report.SourcePreconditionBodyMutationAllowed},
		{"source_proof_graft_allowed", report.SourceProofGraftAllowed},
		{"source_proof_live_admission_enabled", report.SourceProofLiveAdmissionEnabled},
		{"source_proof_body_mutation_allowed", report.SourceProofBodyMutationAllowed},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission decision opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_decision_id", report.WeightedAdmissionResonanceGraftAdmissionDecisionID},
		{"causal_id", report.CausalID},
		{"decision_hash", report.DecisionHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack},
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
			return fmt.Errorf("weighted admission resonance graft admission decision %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema {
		return fmt.Errorf("weighted admission resonance graft admission decision source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionProofPreconditionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_proof_precondition_satisfied_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_proof_precondition_satisfied_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourcePreconditionAction != "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_precondition_action mismatch: got %q want %q", report.SourcePreconditionAction, "consume_weighted_resonance_shadow_graft_admission_proof_before_live_route_admission")
	}
	if report.SourcePreconditionReceiptShape != "weighted_resonance_shadow_graft_admission_proof_precondition_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_precondition_receipt_shape mismatch: got %q want %q", report.SourcePreconditionReceiptShape, "weighted_resonance_shadow_graft_admission_proof_precondition_receipt")
	}
	if report.SourcePreconditionKind != "shadow_graft_admission_proof_precondition" ||
		report.SourcePreconditionMode != "closed_receipt_consumption" ||
		report.SourcePreconditionStage != "pre_live_graft_admission_proof_precondition" {
		return fmt.Errorf("weighted admission resonance graft admission decision source precondition shape mismatch")
	}
	if report.SourcePreconditionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_precondition_body_target mismatch: got %q want %q", report.SourcePreconditionBodyTarget, "none")
	}
	if report.SourceProofAction != "prove_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_proof_action mismatch: got %q want %q", report.SourceProofAction, "prove_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourceProofReceiptShape != "weighted_resonance_shadow_graft_admission_proof_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_proof_receipt_shape mismatch: got %q want %q", report.SourceProofReceiptShape, "weighted_resonance_shadow_graft_admission_proof_receipt")
	}
	if report.SourceProofKind != "shadow_graft_admission_proof" ||
		report.SourceProofMode != "closed_read_back_admission_proof" ||
		report.SourceProofStage != "pre_live_graft_admission_proof" {
		return fmt.Errorf("weighted admission resonance graft admission decision source proof shape mismatch")
	}
	if report.SourceProofBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission decision source_proof_body_target mismatch: got %q want %q", report.SourceProofBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission decision body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-decision-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission decision causal prefix mismatch")
	}
	if !strings.HasPrefix(report.DecisionHash, "weighted-resonance-graft-admission-decision-") {
		return fmt.Errorf("weighted admission resonance graft admission decision hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-decision-read-") ||
		report.DecisionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission decision read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionCausalID, "weighted-resonance-graft-admission-proof-precondition-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash, "weighted-resonance-graft-admission-proof-precondition-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack, "weighted-resonance-graft-admission-proof-precondition-read-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source precondition mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofCausalID, "weighted-resonance-graft-admission-proof-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofHash, "weighted-resonance-graft-admission-proof-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofReadBack, "weighted-resonance-graft-admission-proof-read-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source proof mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission decision source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission decision causal_id mismatch")
	}
	if report.DecisionHash == "" || report.DecisionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission decision decision_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission decision read_back_hash mismatch")
	}
	if report.DecisionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission decision read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionDecisionID == "" || report.WeightedAdmissionResonanceGraftAdmissionDecisionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission decision id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission decision accepted precondition as closed shadow-ready receipt" {
		return fmt.Errorf("weighted admission resonance graft admission decision reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionCausalID(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport) string {
	h := hashJSON(struct {
		SourcePreconditionID   string `json:"source_precondition_id"`
		SourcePreconditionRead string `json:"source_precondition_read_back_hash"`
		SourceProofID          string `json:"source_proof_id"`
		SourceReaderID         string `json:"source_reader_id"`
		Target                 string `json:"target"`
		DecisionKind           string `json:"decision_kind"`
		DecisionStage          string `json:"decision_stage"`
	}{
		SourcePreconditionID:   decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourcePreconditionRead: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack,
		SourceProofID:          decision.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:         decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		Target:                 decision.Target,
		DecisionKind:           decision.DecisionKind,
		DecisionStage:          decision.DecisionStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-decision-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionHash(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport) string {
	h := hashJSON(struct {
		CausalID               string `json:"causal_id"`
		SourcePreconditionID   string `json:"source_precondition_id"`
		SourcePreconditionHash string `json:"source_precondition_hash"`
		SourcePreconditionRead string `json:"source_precondition_read_back_hash"`
		Decision               string `json:"decision"`
		Action                 string `json:"action"`
		ReceiptShape           string `json:"receipt_shape"`
		DecisionMode           string `json:"decision_mode"`
		ProofPrecondition      bool   `json:"proof_precondition_verified"`
		ReadOnly               bool   `json:"read_only"`
		ReplayOnly             bool   `json:"replay_only"`
		AdmissionRequired      bool   `json:"admission_required"`
		ShadowOnly             bool   `json:"shadow_only"`
		DryRunOnly             bool   `json:"dry_run_only"`
		GraftAllowed           bool   `json:"graft_allowed"`
		BodyMutationAllowed    bool   `json:"body_mutation_allowed"`
		LiveAdmissionEnabled   bool   `json:"live_admission_enabled"`
	}{
		CausalID:               decision.CausalID,
		SourcePreconditionID:   decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourcePreconditionHash: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionHash,
		SourcePreconditionRead: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack,
		Decision:               decision.Decision,
		Action:                 decision.Action,
		ReceiptShape:           decision.ReceiptShape,
		DecisionMode:           decision.DecisionMode,
		ProofPrecondition:      decision.ProofPreconditionVerified,
		ReadOnly:               decision.ReadOnly,
		ReplayOnly:             decision.ReplayOnly,
		AdmissionRequired:      decision.AdmissionRequired,
		ShadowOnly:             decision.ShadowOnly,
		DryRunOnly:             decision.DryRunOnly,
		GraftAllowed:           decision.GraftAllowed,
		BodyMutationAllowed:    decision.BodyMutationAllowed,
		LiveAdmissionEnabled:   decision.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-decision-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReadBackHash(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport) string {
	h := hashJSON(struct {
		DecisionHash           string `json:"decision_hash"`
		SourcePreconditionID   string `json:"source_precondition_id"`
		SourcePreconditionRead string `json:"source_precondition_read_back_hash"`
		DecisionKind           string `json:"decision_kind"`
		DecisionReady          bool   `json:"decision_ready"`
		LiveReady              bool   `json:"live_ready"`
		BodyMutation           bool   `json:"body_mutation"`
		LiveAdmission          bool   `json:"live_admission"`
		WriteAllowed           bool   `json:"write_allowed"`
		AdmissionAllowed       bool   `json:"admission_allowed"`
	}{
		DecisionHash:           decision.DecisionHash,
		SourcePreconditionID:   decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourcePreconditionRead: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReadBack,
		DecisionKind:           decision.DecisionKind,
		DecisionReady:          decision.WeightedAdmissionResonanceGraftAdmissionDecisionReady,
		LiveReady:              decision.LiveReady,
		BodyMutation:           decision.BodyMutationAllowed,
		LiveAdmission:          decision.LiveAdmissionEnabled,
		WriteAllowed:           decision.WriteAllowed,
		AdmissionAllowed:       decision.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-decision-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionID(decision admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		Decision                string `json:"decision"`
		SourceReport            string `json:"source_report"`
		SourcePreconditionID    string `json:"source_precondition_id"`
		SourceProofID           string `json:"source_proof_id"`
		SourceReaderID          string `json:"source_reader_id"`
		SourceStoreID           string `json:"source_store_id"`
		SourceCandidateID       string `json:"source_candidate_id"`
		SourceGateID            string `json:"source_gate_id"`
		SourcePreflightID       string `json:"source_preflight_id"`
		SourceBoundaryID        string `json:"source_boundary_id"`
		SourceObservationID     string `json:"source_observation_id"`
		SourceReceiverID        string `json:"source_receiver_id"`
		CausalID                string `json:"causal_id"`
		DecisionHash            string `json:"decision_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		DecisionKind            string `json:"decision_kind"`
		DecisionMode            string `json:"decision_mode"`
		DecisionStage           string `json:"decision_stage"`
		ProofPrecondition       bool   `json:"proof_precondition_verified"`
		AdmissionRequired       bool   `json:"admission_required"`
		ShadowOnly              bool   `json:"shadow_only"`
		GraftAllowed            bool   `json:"graft_allowed"`
		DryRunOnly              bool   `json:"dry_run_only"`
		RawDreamTextAllowed     bool   `json:"raw_dream_text_allowed"`
		JanusSurfaceAllowed     bool   `json:"janus_surface_allowed"`
		CoocLearningAllowed     bool   `json:"cooc_learning_allowed"`
		DeltaHarvestAllowed     bool   `json:"delta_harvest_allowed"`
		BodyMutationAllowed     bool   `json:"body_mutation_allowed"`
		RollbackRequired        bool   `json:"rollback_required"`
		ReadOnly                bool   `json:"read_only"`
		ReplayOnly              bool   `json:"replay_only"`
		LiveReady               bool   `json:"live_ready"`
		ContractsReady          bool   `json:"contracts_ready"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_decision"`
		SourcePreconditionReady bool   `json:"source_precondition_ready"`
		SourceProofReady        bool   `json:"source_proof_ready"`
		SourceReaderReady       bool   `json:"source_reader_ready"`
		SourceStoreReady        bool   `json:"source_store_ready"`
		SourceCandidateReady    bool   `json:"source_candidate_ready"`
		SourceGateReady         bool   `json:"source_gate_ready"`
		SourcePreflightReady    bool   `json:"source_preflight_ready"`
		SourceBoundaryReady     bool   `json:"source_boundary_ready"`
		SourceObservationReady  bool   `json:"source_observation_ready"`
		SourceReceiverReady     bool   `json:"source_receiver_ready"`
		SourceIntentReady       bool   `json:"source_intent_ready"`
		SourceFinalGateReady    bool   `json:"source_final_gate_ready"`
		SourceSealReady         bool   `json:"source_seal_ready"`
		SourcePermitReady       bool   `json:"source_permit_ready"`
		SourceAuthorityUsed     bool   `json:"source_authority_consumed"`
		SourceAuthorityNeeded   bool   `json:"source_authority_required"`
	}{
		Schema:                  decision.Schema,
		Status:                  decision.Status,
		Action:                  decision.Action,
		Decision:                decision.Decision,
		SourceReport:            decision.SourceReport,
		SourcePreconditionID:    decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceProofID:           decision.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:          decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:           decision.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:       decision.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:            decision.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:       decision.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        decision.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     decision.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        decision.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                decision.CausalID,
		DecisionHash:            decision.DecisionHash,
		ReadBackHash:            decision.ReadBackHash,
		Ready:                   decision.WeightedAdmissionResonanceGraftAdmissionDecisionReady,
		ReceiptShape:            decision.ReceiptShape,
		DecisionKind:            decision.DecisionKind,
		DecisionMode:            decision.DecisionMode,
		DecisionStage:           decision.DecisionStage,
		ProofPrecondition:       decision.ProofPreconditionVerified,
		AdmissionRequired:       decision.AdmissionRequired,
		ShadowOnly:              decision.ShadowOnly,
		GraftAllowed:            decision.GraftAllowed,
		DryRunOnly:              decision.DryRunOnly,
		RawDreamTextAllowed:     decision.RawDreamTextAllowed,
		JanusSurfaceAllowed:     decision.JanusSurfaceAllowed,
		CoocLearningAllowed:     decision.CoocLearningAllowed,
		DeltaHarvestAllowed:     decision.DeltaHarvestAllowed,
		BodyMutationAllowed:     decision.BodyMutationAllowed,
		RollbackRequired:        decision.RollbackRequired,
		ReadOnly:                decision.ReadOnly,
		ReplayOnly:              decision.ReplayOnly,
		LiveReady:               decision.LiveReady,
		ContractsReady:          decision.ContractsReady,
		BodyTarget:              decision.BodyTarget,
		WriteAllowed:            decision.WriteAllowed,
		AdmissionAllowed:        decision.AdmissionAllowed,
		LiveAdmissionEnabled:    decision.LiveAdmissionEnabled,
		MutatesState:            decision.MutatesState,
		NextStepBlockedWithout:  decision.NextStepBlockedWithoutResonanceGraftAdmissionDecision,
		SourcePreconditionReady: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceProofReady:        decision.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:       decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:        decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:    decision.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:         decision.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:    decision.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     decision.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  decision.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     decision.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       decision.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    decision.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         decision.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       decision.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:     decision.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:   decision.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-decision-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission decision path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission decision not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission decision not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission decision JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission decision decode failed: %w", err)
	}
	return report, root, nil
}
