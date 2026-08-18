package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_promotion.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport struct {
	Schema                                                               string `json:"schema"`
	Status                                                               string `json:"status"`
	Target                                                               string `json:"target"`
	TargetKind                                                           string `json:"target_kind"`
	TargetMode                                                           string `json:"target_mode"`
	Action                                                               string `json:"action"`
	Promotion                                                            string `json:"promotion"`
	WeightedAdmissionResonanceGraftAdmissionPromotionReady               bool   `json:"weighted_admission_resonance_graft_admission_promotion_ready"`
	WeightedAdmissionResonanceGraftAdmissionDecisionConsumed             bool   `json:"weighted_admission_resonance_graft_admission_decision_consumed"`
	WeightedAdmissionResonanceGraftAdmissionDecisionRequired             bool   `json:"weighted_admission_resonance_graft_admission_decision_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionPromotion               bool   `json:"next_step_blocked_without_resonance_graft_admission_promotion"`
	WeightedAdmissionResonanceGraftAdmissionPromotionID                  string `json:"weighted_admission_resonance_graft_admission_promotion_id"`
	ReceiptShape                                                         string `json:"receipt_shape"`
	PromotionKind                                                        string `json:"promotion_kind"`
	PromotionMode                                                        string `json:"promotion_mode"`
	PromotionStage                                                       string `json:"promotion_stage"`
	CausalID                                                             string `json:"causal_id"`
	PromotionHash                                                        string `json:"promotion_hash"`
	ReadBackHash                                                         string `json:"read_back_hash"`
	DecisionVerified                                                     bool   `json:"decision_verified"`
	DecisionHashVerified                                                 bool   `json:"decision_hash_verified"`
	DecisionReadBackVerified                                             bool   `json:"decision_read_back_verified"`
	ProofPreconditionVerified                                            bool   `json:"proof_precondition_verified"`
	PreconditionHashVerified                                             bool   `json:"precondition_hash_verified"`
	PreconditionReadBackVerified                                         bool   `json:"precondition_read_back_verified"`
	ProofVerified                                                        bool   `json:"proof_verified"`
	ProofHashVerified                                                    bool   `json:"proof_hash_verified"`
	ProofReadBackVerified                                                bool   `json:"proof_read_back_verified"`
	StoreReaderVerified                                                  bool   `json:"store_reader_verified"`
	StoreVerified                                                        bool   `json:"store_verified"`
	CandidateVerified                                                    bool   `json:"candidate_verified"`
	GateVerified                                                         bool   `json:"gate_verified"`
	PreflightVerified                                                    bool   `json:"preflight_verified"`
	BoundaryVerified                                                     bool   `json:"boundary_verified"`
	ObservationVerified                                                  bool   `json:"observation_verified"`
	ReceiverVerified                                                     bool   `json:"receiver_verified"`
	IntentVerified                                                       bool   `json:"intent_verified"`
	FinalGateVerified                                                    bool   `json:"final_gate_verified"`
	SealVerified                                                         bool   `json:"seal_verified"`
	PermitVerified                                                       bool   `json:"permit_verified"`
	AuthorityVerified                                                    bool   `json:"authority_verified"`
	AdmissionRequired                                                    bool   `json:"admission_required"`
	ShadowOnly                                                           bool   `json:"shadow_only"`
	GraftAllowed                                                         bool   `json:"graft_allowed"`
	DryRunOnly                                                           bool   `json:"dry_run_only"`
	LiveReady                                                            bool   `json:"live_ready"`
	RawDreamTextAllowed                                                  bool   `json:"raw_dream_text_allowed"`
	RawDreamTextObserved                                                 bool   `json:"raw_dream_text_observed"`
	RawDreamTextForwarded                                                bool   `json:"raw_dream_text_forwarded"`
	JanusSurfaceAllowed                                                  bool   `json:"janus_surface_allowed"`
	CoocLearningAllowed                                                  bool   `json:"cooc_learning_allowed"`
	DeltaHarvestAllowed                                                  bool   `json:"delta_harvest_allowed"`
	BodyMutationAllowed                                                  bool   `json:"body_mutation_allowed"`
	RollbackRequired                                                     bool   `json:"rollback_required"`
	ReadOnly                                                             bool   `json:"read_only"`
	ReplayOnly                                                           bool   `json:"replay_only"`
	SourceSchema                                                         string `json:"source_schema"`
	SourceStatus                                                         string `json:"source_status"`
	SourceTarget                                                         string `json:"source_target"`
	SourceReport                                                         string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionID             string `json:"source_weighted_admission_resonance_graft_admission_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady          bool   `json:"source_weighted_admission_resonance_graft_admission_decision_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionCausalID       string `json:"source_weighted_admission_resonance_graft_admission_decision_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash           string `json:"source_weighted_admission_resonance_graft_admission_decision_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack       string `json:"source_weighted_admission_resonance_graft_admission_decision_read_back_hash"`
	SourceDecision                                                       string `json:"source_decision"`
	SourceDecisionAction                                                 string `json:"source_decision_action"`
	SourceDecisionReceiptShape                                           string `json:"source_decision_receipt_shape"`
	SourceDecisionKind                                                   string `json:"source_decision_kind"`
	SourceDecisionMode                                                   string `json:"source_decision_mode"`
	SourceDecisionStage                                                  string `json:"source_decision_stage"`
	SourceDecisionAdmissionRequired                                      bool   `json:"source_decision_admission_required"`
	SourceDecisionShadowOnly                                             bool   `json:"source_decision_shadow_only"`
	SourceDecisionGraftAllowed                                           bool   `json:"source_decision_graft_allowed"`
	SourceDecisionDryRunOnly                                             bool   `json:"source_decision_dry_run_only"`
	SourceDecisionLiveReady                                              bool   `json:"source_decision_live_ready"`
	SourceDecisionRawDreamTextAllowed                                    bool   `json:"source_decision_raw_dream_text_allowed"`
	SourceDecisionRawDreamTextObserved                                   bool   `json:"source_decision_raw_dream_text_observed"`
	SourceDecisionRawDreamTextForwarded                                  bool   `json:"source_decision_raw_dream_text_forwarded"`
	SourceDecisionJanusSurfaceAllowed                                    bool   `json:"source_decision_janus_surface_allowed"`
	SourceDecisionCoocLearningAllowed                                    bool   `json:"source_decision_cooc_learning_allowed"`
	SourceDecisionDeltaHarvestAllowed                                    bool   `json:"source_decision_delta_harvest_allowed"`
	SourceDecisionBodyMutationAllowed                                    bool   `json:"source_decision_body_mutation_allowed"`
	SourceDecisionRollbackRequired                                       bool   `json:"source_decision_rollback_required"`
	SourceDecisionReadOnly                                               bool   `json:"source_decision_read_only"`
	SourceDecisionReplayOnly                                             bool   `json:"source_decision_replay_only"`
	SourceDecisionWriteAllowed                                           bool   `json:"source_decision_write_allowed"`
	SourceDecisionAdmissionAllowed                                       bool   `json:"source_decision_admission_allowed"`
	SourceDecisionLiveAdmissionEnabled                                   bool   `json:"source_decision_live_admission_enabled"`
	SourceDecisionMutatesState                                           bool   `json:"source_decision_mutates_state"`
	SourceDecisionBodyTarget                                             string `json:"source_decision_body_target"`
	SourceDecisionPassed                                                 bool   `json:"source_decision_passed"`
	SourceDecisionReason                                                 string `json:"source_decision_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID    string `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady bool   `json:"source_weighted_admission_resonance_graft_admission_proof_precondition_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofID                string `json:"source_weighted_admission_resonance_graft_admission_proof_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionProofReady             bool   `json:"source_weighted_admission_resonance_graft_admission_proof_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID          string `json:"source_weighted_admission_resonance_graft_candidate_store_reader_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady       bool   `json:"source_weighted_admission_resonance_graft_candidate_store_reader_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreID                string `json:"source_weighted_admission_resonance_graft_candidate_store_id"`
	SourceWeightedAdmissionResonanceGraftCandidateStoreReady             bool   `json:"source_weighted_admission_resonance_graft_candidate_store_ready"`
	SourceWeightedAdmissionResonanceGraftCandidateID                     string `json:"source_weighted_admission_resonance_graft_candidate_id"`
	SourceWeightedAdmissionResonanceGraftCandidateReady                  bool   `json:"source_weighted_admission_resonance_graft_candidate_ready"`
	SourceWeightedAdmissionResonanceGraftGateID                          string `json:"source_weighted_admission_resonance_graft_gate_id"`
	SourceWeightedAdmissionResonanceGraftGateReady                       bool   `json:"source_weighted_admission_resonance_graft_gate_ready"`
	SourceWeightedAdmissionResonanceGraftPreflightID                     string `json:"source_weighted_admission_resonance_graft_preflight_id"`
	SourceWeightedAdmissionResonanceGraftPreflightReady                  bool   `json:"source_weighted_admission_resonance_graft_preflight_ready"`
	SourceWeightedAdmissionResonanceGraftBoundaryID                      string `json:"source_weighted_admission_resonance_graft_boundary_id"`
	SourceWeightedAdmissionResonanceGraftBoundaryReady                   bool   `json:"source_weighted_admission_resonance_graft_boundary_ready"`
	SourceWeightedAdmissionResonanceObservationID                        string `json:"source_weighted_admission_resonance_observation_id"`
	SourceWeightedAdmissionResonanceObservationReady                     bool   `json:"source_weighted_admission_resonance_observation_ready"`
	SourceWeightedAdmissionResonanceReceiverID                           string `json:"source_weighted_admission_resonance_receiver_id"`
	SourceWeightedAdmissionResonanceReceiverReady                        bool   `json:"source_weighted_admission_resonance_receiver_ready"`
	SourceWeightedAdmissionResonanceIntentReady                          bool   `json:"source_weighted_admission_resonance_intent_ready"`
	SourceWeightedAdmissionFinalGateReady                                bool   `json:"source_weighted_admission_final_gate_ready"`
	SourceWeightedAdmissionSealReady                                     bool   `json:"source_weighted_admission_seal_ready"`
	SourceWeightedAdmissionPermitReady                                   bool   `json:"source_weighted_admission_permit_ready"`
	SourceWeightedAdmissionAuthorityConsumed                             bool   `json:"source_weighted_admission_authority_consumed"`
	SourceWeightedAdmissionAuthorityRequired                             bool   `json:"source_weighted_admission_authority_required"`
	BodySmokeWeighted                                                    bool   `json:"body_smoke_weighted"`
	NanoDirectRunner                                                     bool   `json:"nano_direct_runner"`
	NanoDirectFinalGate                                                  bool   `json:"nano_direct_final_gate"`
	ResonanceGraftAdmissionProof                                         bool   `json:"resonance_graft_admission_proof"`
	BoundaryReportFullChain                                              bool   `json:"boundary_report_full_chain"`
	SourceAuthorityGranted                                               bool   `json:"source_authority_granted"`
	AuthorityGranted                                                     bool   `json:"authority_granted"`
	ContractsReady                                                       bool   `json:"contracts_ready"`
	WriteAllowed                                                         bool   `json:"write_allowed"`
	AdmissionAllowed                                                     bool   `json:"admission_allowed"`
	LiveAdmissionEnabled                                                 bool   `json:"live_admission_enabled"`
	MutatesState                                                         bool   `json:"mutates_state"`
	BodyTarget                                                           string `json:"body_target"`
	Passed                                                               bool   `json:"passed"`
	Reason                                                               string `json:"reason"`
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotion(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion RESONANCE_GRAFT_ADMISSION_DECISION_REPORT RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT")
	}
	decisionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission promotion output path missing")
	}
	decision, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportForAssert(decisionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionReportError(decision, root); err != nil {
		return err
	}
	promotion := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport{
		Schema:         admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema,
		Status:         "shadow_graft_admission_promotion_ready_dry_run",
		Target:         "live_route_admission_next_step",
		TargetKind:     "weighted_internal_world_shadow_graft_admission_promotion",
		TargetMode:     "closed_promotion_receipt_dry_run",
		Action:         "promote_weighted_resonance_shadow_graft_admission_dry_run",
		Promotion:      "pending_live_admission",
		ReceiptShape:   "weighted_resonance_shadow_graft_admission_promotion_receipt",
		PromotionKind:  "shadow_graft_admission_promotion",
		PromotionMode:  "closed_decision_promotion",
		PromotionStage: "pre_live_graft_admission_promotion",
		WeightedAdmissionResonanceGraftAdmissionPromotionReady:   true,
		WeightedAdmissionResonanceGraftAdmissionDecisionConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionDecisionRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionPromotion:   true,
		DecisionVerified:             true,
		DecisionHashVerified:         true,
		DecisionReadBackVerified:     true,
		ProofPreconditionVerified:    decision.ProofPreconditionVerified,
		PreconditionHashVerified:     decision.PreconditionHashVerified,
		PreconditionReadBackVerified: decision.PreconditionReadBackVerified,
		ProofVerified:                decision.ProofVerified,
		ProofHashVerified:            decision.ProofHashVerified,
		ProofReadBackVerified:        decision.ProofReadBackVerified,
		StoreReaderVerified:          decision.StoreReaderVerified,
		StoreVerified:                decision.StoreVerified,
		CandidateVerified:            decision.CandidateVerified,
		GateVerified:                 decision.GateVerified,
		PreflightVerified:            decision.PreflightVerified,
		BoundaryVerified:             decision.BoundaryVerified,
		ObservationVerified:          decision.ObservationVerified,
		ReceiverVerified:             decision.ReceiverVerified,
		IntentVerified:               decision.IntentVerified,
		FinalGateVerified:            decision.FinalGateVerified,
		SealVerified:                 decision.SealVerified,
		PermitVerified:               decision.PermitVerified,
		AuthorityVerified:            decision.AuthorityVerified,
		AdmissionRequired:            true,
		ShadowOnly:                   true,
		GraftAllowed:                 false,
		DryRunOnly:                   true,
		LiveReady:                    true,
		RawDreamTextAllowed:          false,
		RawDreamTextObserved:         false,
		RawDreamTextForwarded:        false,
		JanusSurfaceAllowed:          false,
		CoocLearningAllowed:          false,
		DeltaHarvestAllowed:          false,
		BodyMutationAllowed:          false,
		RollbackRequired:             true,
		ReadOnly:                     true,
		ReplayOnly:                   true,
		SourceSchema:                 decision.Schema,
		SourceStatus:                 decision.Status,
		SourceTarget:                 decision.Target,
		SourceReport:                 decisionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:       decision.WeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:    decision.WeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionCausalID: decision.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash:     decision.DecisionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack: decision.ReadBackHash,
		SourceDecision:                      decision.Decision,
		SourceDecisionAction:                decision.Action,
		SourceDecisionReceiptShape:          decision.ReceiptShape,
		SourceDecisionKind:                  decision.DecisionKind,
		SourceDecisionMode:                  decision.DecisionMode,
		SourceDecisionStage:                 decision.DecisionStage,
		SourceDecisionAdmissionRequired:     decision.AdmissionRequired,
		SourceDecisionShadowOnly:            decision.ShadowOnly,
		SourceDecisionGraftAllowed:          decision.GraftAllowed,
		SourceDecisionDryRunOnly:            decision.DryRunOnly,
		SourceDecisionLiveReady:             decision.LiveReady,
		SourceDecisionRawDreamTextAllowed:   decision.RawDreamTextAllowed,
		SourceDecisionRawDreamTextObserved:  decision.RawDreamTextObserved,
		SourceDecisionRawDreamTextForwarded: decision.RawDreamTextForwarded,
		SourceDecisionJanusSurfaceAllowed:   decision.JanusSurfaceAllowed,
		SourceDecisionCoocLearningAllowed:   decision.CoocLearningAllowed,
		SourceDecisionDeltaHarvestAllowed:   decision.DeltaHarvestAllowed,
		SourceDecisionBodyMutationAllowed:   decision.BodyMutationAllowed,
		SourceDecisionRollbackRequired:      decision.RollbackRequired,
		SourceDecisionReadOnly:              decision.ReadOnly,
		SourceDecisionReplayOnly:            decision.ReplayOnly,
		SourceDecisionWriteAllowed:          decision.WriteAllowed,
		SourceDecisionAdmissionAllowed:      decision.AdmissionAllowed,
		SourceDecisionLiveAdmissionEnabled:  decision.LiveAdmissionEnabled,
		SourceDecisionMutatesState:          decision.MutatesState,
		SourceDecisionBodyTarget:            decision.BodyTarget,
		SourceDecisionPassed:                decision.Passed,
		SourceDecisionReason:                decision.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:    decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: decision.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                decision.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:             decision.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:          decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:       decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                decision.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:             decision.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                     decision.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                  decision.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                          decision.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                       decision.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                     decision.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                  decision.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                      decision.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                   decision.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                        decision.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                     decision.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                           decision.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                        decision.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                          decision.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                decision.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                     decision.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                   decision.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                             decision.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                             decision.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                    decision.BodySmokeWeighted,
		NanoDirectRunner:                                                     decision.NanoDirectRunner,
		NanoDirectFinalGate:                                                  decision.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                         decision.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                              decision.BoundaryReportFullChain,
		SourceAuthorityGranted:                                               decision.SourceAuthorityGranted,
		AuthorityGranted:                                                     false,
		ContractsReady:                                                       false,
		WriteAllowed:                                                         false,
		AdmissionAllowed:                                                     false,
		LiveAdmissionEnabled:                                                 false,
		MutatesState:                                                         false,
		BodyTarget:                                                           "none",
		Passed:                                                               true,
		Reason:                                                               "weighted resonance shadow graft admission decision promoted as pending live admission while closed",
	}
	promotion.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionCausalID(promotion)
	promotion.PromotionHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionHash(promotion)
	promotion.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReadBackHash(promotion)
	promotion.WeightedAdmissionResonanceGraftAdmissionPromotionID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionID(promotion)
	if promotion.CausalID == "" ||
		promotion.PromotionHash == "" ||
		promotion.ReadBackHash == "" ||
		promotion.WeightedAdmissionResonanceGraftAdmissionPromotionID == "" ||
		promotion.PromotionHash == promotion.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission promotion read-back proof failed")
	}
	raw, err := json.MarshalIndent(promotion, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission promotion marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission promotion write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-promotion] pass: resonance_graft_admission_promotion_report=%s resonance_graft_admission_decision_report=%s\n", outputPath, decisionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-promotion-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission promotion schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema {
		return fmt.Errorf("weighted admission resonance graft admission promotion schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema)
	}
	if report.Status != "shadow_graft_admission_promotion_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission promotion status mismatch: got %q want %q", report.Status, "shadow_graft_admission_promotion_ready_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission promotion target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission promotion target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_promotion")
	}
	if report.TargetMode != "closed_promotion_receipt_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission promotion target_mode mismatch: got %q want %q", report.TargetMode, "closed_promotion_receipt_dry_run")
	}
	if report.Action != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission promotion action mismatch: got %q want %q", report.Action, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission promotion promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission promotion receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_promotion_receipt")
	}
	if report.PromotionKind != "shadow_graft_admission_promotion" ||
		report.PromotionMode != "closed_decision_promotion" ||
		report.PromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission promotion shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_promotion_ready", report.WeightedAdmissionResonanceGraftAdmissionPromotionReady},
		{"weighted_admission_resonance_graft_admission_decision_consumed", report.WeightedAdmissionResonanceGraftAdmissionDecisionConsumed},
		{"weighted_admission_resonance_graft_admission_decision_required", report.WeightedAdmissionResonanceGraftAdmissionDecisionRequired},
		{"next_step_blocked_without_resonance_graft_admission_promotion", report.NextStepBlockedWithoutResonanceGraftAdmissionPromotion},
		{"decision_verified", report.DecisionVerified},
		{"decision_hash_verified", report.DecisionHashVerified},
		{"decision_read_back_verified", report.DecisionReadBackVerified},
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
		{"source_weighted_admission_resonance_graft_admission_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady},
		{"source_decision_admission_required", report.SourceDecisionAdmissionRequired},
		{"source_decision_shadow_only", report.SourceDecisionShadowOnly},
		{"source_decision_dry_run_only", report.SourceDecisionDryRunOnly},
		{"source_decision_live_ready", report.SourceDecisionLiveReady},
		{"source_decision_rollback_required", report.SourceDecisionRollbackRequired},
		{"source_decision_read_only", report.SourceDecisionReadOnly},
		{"source_decision_replay_only", report.SourceDecisionReplayOnly},
		{"source_decision_passed", report.SourceDecisionPassed},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady},
		{"source_weighted_admission_resonance_graft_admission_proof_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionProofReady},
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
			return fmt.Errorf("weighted admission resonance graft admission promotion %s not ready", required.name)
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
		{"source_decision_graft_allowed", report.SourceDecisionGraftAllowed},
		{"source_decision_raw_dream_text_allowed", report.SourceDecisionRawDreamTextAllowed},
		{"source_decision_raw_dream_text_observed", report.SourceDecisionRawDreamTextObserved},
		{"source_decision_raw_dream_text_forwarded", report.SourceDecisionRawDreamTextForwarded},
		{"source_decision_janus_surface_allowed", report.SourceDecisionJanusSurfaceAllowed},
		{"source_decision_cooc_learning_allowed", report.SourceDecisionCoocLearningAllowed},
		{"source_decision_delta_harvest_allowed", report.SourceDecisionDeltaHarvestAllowed},
		{"source_decision_body_mutation_allowed", report.SourceDecisionBodyMutationAllowed},
		{"source_decision_write_allowed", report.SourceDecisionWriteAllowed},
		{"source_decision_admission_allowed", report.SourceDecisionAdmissionAllowed},
		{"source_decision_live_admission_enabled", report.SourceDecisionLiveAdmissionEnabled},
		{"source_decision_mutates_state", report.SourceDecisionMutatesState},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission promotion opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_promotion_id", report.WeightedAdmissionResonanceGraftAdmissionPromotionID},
		{"causal_id", report.CausalID},
		{"promotion_hash", report.PromotionHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID},
		{"source_weighted_admission_resonance_graft_admission_decision_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionCausalID},
		{"source_weighted_admission_resonance_graft_admission_decision_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash},
		{"source_weighted_admission_resonance_graft_admission_decision_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack},
		{"source_weighted_admission_resonance_graft_admission_proof_precondition_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID},
		{"source_weighted_admission_resonance_graft_admission_proof_id", report.SourceWeightedAdmissionResonanceGraftAdmissionProofID},
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
			return fmt.Errorf("weighted admission resonance graft admission promotion %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionDecisionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_decision_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_decision_ready_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceDecision != "shadow_ready" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_decision mismatch: got %q want %q", report.SourceDecision, "shadow_ready")
	}
	if report.SourceDecisionAction != "decide_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_decision_action mismatch: got %q want %q", report.SourceDecisionAction, "decide_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourceDecisionReceiptShape != "weighted_resonance_shadow_graft_admission_decision_receipt" ||
		report.SourceDecisionKind != "shadow_graft_admission_decision" ||
		report.SourceDecisionMode != "closed_precondition_decision" ||
		report.SourceDecisionStage != "pre_live_graft_admission_decision" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source decision shape mismatch")
	}
	if report.SourceDecisionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission promotion source_decision_body_target mismatch: got %q want %q", report.SourceDecisionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission promotion body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-promotion-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion causal prefix mismatch")
	}
	if !strings.HasPrefix(report.PromotionHash, "weighted-resonance-graft-admission-promotion-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-promotion-read-") ||
		report.PromotionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission promotion read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionCausalID, "weighted-resonance-graft-admission-decision-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash, "weighted-resonance-graft-admission-decision-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack, "weighted-resonance-graft-admission-decision-read-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source decision mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission promotion source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission promotion causal_id mismatch")
	}
	if report.PromotionHash == "" || report.PromotionHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission promotion promotion_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission promotion read_back_hash mismatch")
	}
	if report.PromotionHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission promotion read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionPromotionID == "" || report.WeightedAdmissionResonanceGraftAdmissionPromotionID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionID(report) {
		return fmt.Errorf("weighted admission resonance graft admission promotion id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission decision promoted as pending live admission while closed" {
		return fmt.Errorf("weighted admission resonance graft admission promotion reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionCausalID(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport) string {
	h := hashJSON(struct {
		SourceDecisionID   string `json:"source_decision_id"`
		SourceDecisionRead string `json:"source_decision_read_back_hash"`
		SourceProofID      string `json:"source_proof_id"`
		SourceReaderID     string `json:"source_reader_id"`
		Target             string `json:"target"`
		PromotionKind      string `json:"promotion_kind"`
		PromotionStage     string `json:"promotion_stage"`
	}{
		SourceDecisionID:   promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceDecisionRead: promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack,
		SourceProofID:      promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:     promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		Target:             promotion.Target,
		PromotionKind:      promotion.PromotionKind,
		PromotionStage:     promotion.PromotionStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-promotion-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionHash(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport) string {
	h := hashJSON(struct {
		CausalID           string `json:"causal_id"`
		SourceDecisionID   string `json:"source_decision_id"`
		SourceDecisionHash string `json:"source_decision_hash"`
		SourceDecisionRead string `json:"source_decision_read_back_hash"`
		Promotion          string `json:"promotion"`
		Action             string `json:"action"`
		ReceiptShape       string `json:"receipt_shape"`
		PromotionMode      string `json:"promotion_mode"`
		DecisionVerified   bool   `json:"decision_verified"`
		ReadOnly           bool   `json:"read_only"`
		ReplayOnly         bool   `json:"replay_only"`
		AdmissionRequired  bool   `json:"admission_required"`
		ShadowOnly         bool   `json:"shadow_only"`
		DryRunOnly         bool   `json:"dry_run_only"`
		GraftAllowed       bool   `json:"graft_allowed"`
		BodyMutation       bool   `json:"body_mutation_allowed"`
		LiveAdmission      bool   `json:"live_admission_enabled"`
	}{
		CausalID:           promotion.CausalID,
		SourceDecisionID:   promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceDecisionHash: promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionHash,
		SourceDecisionRead: promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack,
		Promotion:          promotion.Promotion,
		Action:             promotion.Action,
		ReceiptShape:       promotion.ReceiptShape,
		PromotionMode:      promotion.PromotionMode,
		DecisionVerified:   promotion.DecisionVerified,
		ReadOnly:           promotion.ReadOnly,
		ReplayOnly:         promotion.ReplayOnly,
		AdmissionRequired:  promotion.AdmissionRequired,
		ShadowOnly:         promotion.ShadowOnly,
		DryRunOnly:         promotion.DryRunOnly,
		GraftAllowed:       promotion.GraftAllowed,
		BodyMutation:       promotion.BodyMutationAllowed,
		LiveAdmission:      promotion.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-promotion-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReadBackHash(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport) string {
	h := hashJSON(struct {
		PromotionHash      string `json:"promotion_hash"`
		SourceDecisionID   string `json:"source_decision_id"`
		SourceDecisionRead string `json:"source_decision_read_back_hash"`
		PromotionKind      string `json:"promotion_kind"`
		PromotionReady     bool   `json:"promotion_ready"`
		DecisionConsumed   bool   `json:"decision_consumed"`
		LiveReady          bool   `json:"live_ready"`
		BodyMutation       bool   `json:"body_mutation"`
		LiveAdmission      bool   `json:"live_admission"`
		WriteAllowed       bool   `json:"write_allowed"`
		AdmissionAllowed   bool   `json:"admission_allowed"`
	}{
		PromotionHash:      promotion.PromotionHash,
		SourceDecisionID:   promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceDecisionRead: promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReadBack,
		PromotionKind:      promotion.PromotionKind,
		PromotionReady:     promotion.WeightedAdmissionResonanceGraftAdmissionPromotionReady,
		DecisionConsumed:   promotion.WeightedAdmissionResonanceGraftAdmissionDecisionConsumed,
		LiveReady:          promotion.LiveReady,
		BodyMutation:       promotion.BodyMutationAllowed,
		LiveAdmission:      promotion.LiveAdmissionEnabled,
		WriteAllowed:       promotion.WriteAllowed,
		AdmissionAllowed:   promotion.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-promotion-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionID(promotion admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		Promotion               string `json:"promotion"`
		SourceReport            string `json:"source_report"`
		SourceDecisionID        string `json:"source_decision_id"`
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
		PromotionHash           string `json:"promotion_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		PromotionKind           string `json:"promotion_kind"`
		PromotionMode           string `json:"promotion_mode"`
		PromotionStage          string `json:"promotion_stage"`
		DecisionVerified        bool   `json:"decision_verified"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_promotion"`
		SourceDecisionReady     bool   `json:"source_decision_ready"`
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
		Schema:                  promotion.Schema,
		Status:                  promotion.Status,
		Action:                  promotion.Action,
		Promotion:               promotion.Promotion,
		SourceReport:            promotion.SourceReport,
		SourceDecisionID:        promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceProofID:           promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:          promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:           promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:       promotion.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:            promotion.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:       promotion.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        promotion.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     promotion.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        promotion.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                promotion.CausalID,
		PromotionHash:           promotion.PromotionHash,
		ReadBackHash:            promotion.ReadBackHash,
		Ready:                   promotion.WeightedAdmissionResonanceGraftAdmissionPromotionReady,
		ReceiptShape:            promotion.ReceiptShape,
		PromotionKind:           promotion.PromotionKind,
		PromotionMode:           promotion.PromotionMode,
		PromotionStage:          promotion.PromotionStage,
		DecisionVerified:        promotion.DecisionVerified,
		AdmissionRequired:       promotion.AdmissionRequired,
		ShadowOnly:              promotion.ShadowOnly,
		GraftAllowed:            promotion.GraftAllowed,
		DryRunOnly:              promotion.DryRunOnly,
		RawDreamTextAllowed:     promotion.RawDreamTextAllowed,
		JanusSurfaceAllowed:     promotion.JanusSurfaceAllowed,
		CoocLearningAllowed:     promotion.CoocLearningAllowed,
		DeltaHarvestAllowed:     promotion.DeltaHarvestAllowed,
		BodyMutationAllowed:     promotion.BodyMutationAllowed,
		RollbackRequired:        promotion.RollbackRequired,
		ReadOnly:                promotion.ReadOnly,
		ReplayOnly:              promotion.ReplayOnly,
		LiveReady:               promotion.LiveReady,
		ContractsReady:          promotion.ContractsReady,
		BodyTarget:              promotion.BodyTarget,
		WriteAllowed:            promotion.WriteAllowed,
		AdmissionAllowed:        promotion.AdmissionAllowed,
		LiveAdmissionEnabled:    promotion.LiveAdmissionEnabled,
		MutatesState:            promotion.MutatesState,
		NextStepBlockedWithout:  promotion.NextStepBlockedWithoutResonanceGraftAdmissionPromotion,
		SourceDecisionReady:     promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourcePreconditionReady: promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceProofReady:        promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:       promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:        promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:    promotion.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:         promotion.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:    promotion.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     promotion.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  promotion.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     promotion.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       promotion.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    promotion.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         promotion.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       promotion.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:     promotion.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:   promotion.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-promotion-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission promotion path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission promotion not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission promotion not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission promotion JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission promotion decode failed: %w", err)
	}
	return report, root, nil
}
