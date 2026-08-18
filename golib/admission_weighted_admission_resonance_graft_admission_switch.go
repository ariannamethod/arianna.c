package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport struct {
	Schema                                                               string `json:"schema"`
	Status                                                               string `json:"status"`
	Target                                                               string `json:"target"`
	TargetKind                                                           string `json:"target_kind"`
	TargetMode                                                           string `json:"target_mode"`
	Action                                                               string `json:"action"`
	SwitchState                                                          string `json:"switch_state"`
	SwitchAction                                                         string `json:"switch_action"`
	Promotion                                                            string `json:"promotion"`
	WeightedAdmissionResonanceGraftAdmissionSwitchReady                  bool   `json:"weighted_admission_resonance_graft_admission_switch_ready"`
	WeightedAdmissionResonanceGraftAdmissionPromotionConsumed            bool   `json:"weighted_admission_resonance_graft_admission_promotion_consumed"`
	WeightedAdmissionResonanceGraftAdmissionPromotionRequired            bool   `json:"weighted_admission_resonance_graft_admission_promotion_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionSwitch                  bool   `json:"next_step_blocked_without_resonance_graft_admission_switch"`
	WeightedAdmissionResonanceGraftAdmissionSwitchID                     string `json:"weighted_admission_resonance_graft_admission_switch_id"`
	ReceiptShape                                                         string `json:"receipt_shape"`
	SwitchKind                                                           string `json:"switch_kind"`
	SwitchMode                                                           string `json:"switch_mode"`
	SwitchStage                                                          string `json:"switch_stage"`
	CausalID                                                             string `json:"causal_id"`
	SwitchHash                                                           string `json:"switch_hash"`
	ReadBackHash                                                         string `json:"read_back_hash"`
	PromotionVerified                                                    bool   `json:"promotion_verified"`
	PromotionHashVerified                                                bool   `json:"promotion_hash_verified"`
	PromotionReadBackVerified                                            bool   `json:"promotion_read_back_verified"`
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
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionID            string `json:"source_weighted_admission_resonance_graft_admission_promotion_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady         bool   `json:"source_weighted_admission_resonance_graft_admission_promotion_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID      string `json:"source_weighted_admission_resonance_graft_admission_promotion_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash          string `json:"source_weighted_admission_resonance_graft_admission_promotion_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack      string `json:"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash"`
	SourcePromotion                                                      string `json:"source_promotion"`
	SourcePromotionAction                                                string `json:"source_promotion_action"`
	SourcePromotionReceiptShape                                          string `json:"source_promotion_receipt_shape"`
	SourcePromotionKind                                                  string `json:"source_promotion_kind"`
	SourcePromotionMode                                                  string `json:"source_promotion_mode"`
	SourcePromotionStage                                                 string `json:"source_promotion_stage"`
	SourcePromotionAdmissionRequired                                     bool   `json:"source_promotion_admission_required"`
	SourcePromotionShadowOnly                                            bool   `json:"source_promotion_shadow_only"`
	SourcePromotionGraftAllowed                                          bool   `json:"source_promotion_graft_allowed"`
	SourcePromotionDryRunOnly                                            bool   `json:"source_promotion_dry_run_only"`
	SourcePromotionLiveReady                                             bool   `json:"source_promotion_live_ready"`
	SourcePromotionRawDreamTextAllowed                                   bool   `json:"source_promotion_raw_dream_text_allowed"`
	SourcePromotionRawDreamTextObserved                                  bool   `json:"source_promotion_raw_dream_text_observed"`
	SourcePromotionRawDreamTextForwarded                                 bool   `json:"source_promotion_raw_dream_text_forwarded"`
	SourcePromotionJanusSurfaceAllowed                                   bool   `json:"source_promotion_janus_surface_allowed"`
	SourcePromotionCoocLearningAllowed                                   bool   `json:"source_promotion_cooc_learning_allowed"`
	SourcePromotionDeltaHarvestAllowed                                   bool   `json:"source_promotion_delta_harvest_allowed"`
	SourcePromotionBodyMutationAllowed                                   bool   `json:"source_promotion_body_mutation_allowed"`
	SourcePromotionRollbackRequired                                      bool   `json:"source_promotion_rollback_required"`
	SourcePromotionReadOnly                                              bool   `json:"source_promotion_read_only"`
	SourcePromotionReplayOnly                                            bool   `json:"source_promotion_replay_only"`
	SourcePromotionWriteAllowed                                          bool   `json:"source_promotion_write_allowed"`
	SourcePromotionAdmissionAllowed                                      bool   `json:"source_promotion_admission_allowed"`
	SourcePromotionLiveAdmissionEnabled                                  bool   `json:"source_promotion_live_admission_enabled"`
	SourcePromotionMutatesState                                          bool   `json:"source_promotion_mutates_state"`
	SourcePromotionBodyTarget                                            string `json:"source_promotion_body_target"`
	SourcePromotionPassed                                                bool   `json:"source_promotion_passed"`
	SourcePromotionReason                                                string `json:"source_promotion_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionID             string `json:"source_weighted_admission_resonance_graft_admission_decision_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady          bool   `json:"source_weighted_admission_resonance_graft_admission_decision_ready"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitch(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch RESONANCE_GRAFT_ADMISSION_PROMOTION_REPORT RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT")
	}
	promotionPath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission switch output path missing")
	}
	promotion, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportForAssert(promotionPath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionReportError(promotion, root); err != nil {
		return err
	}
	sw := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport{
		Schema:       admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema,
		Status:       "shadow_graft_admission_switch_disabled_dry_run",
		Target:       "live_route_admission_next_step",
		TargetKind:   "weighted_internal_world_shadow_graft_admission_switch",
		TargetMode:   "closed_switch_guard_dry_run",
		Action:       "hold_weighted_resonance_shadow_graft_admission_promotion_dry_run",
		SwitchState:  "disabled",
		SwitchAction: "hold_pending_live_admission",
		Promotion:    "pending_live_admission",
		ReceiptShape: "weighted_resonance_shadow_graft_admission_switch_receipt",
		SwitchKind:   "shadow_graft_admission_switch",
		SwitchMode:   "closed_promotion_switch_guard",
		SwitchStage:  "pre_live_graft_admission_switch",
		WeightedAdmissionResonanceGraftAdmissionSwitchReady:       true,
		WeightedAdmissionResonanceGraftAdmissionPromotionConsumed: true,
		WeightedAdmissionResonanceGraftAdmissionPromotionRequired: true,
		NextStepBlockedWithoutResonanceGraftAdmissionSwitch:       true,
		PromotionVerified:            true,
		PromotionHashVerified:        true,
		PromotionReadBackVerified:    true,
		DecisionVerified:             promotion.DecisionVerified,
		DecisionHashVerified:         promotion.DecisionHashVerified,
		DecisionReadBackVerified:     promotion.DecisionReadBackVerified,
		ProofPreconditionVerified:    promotion.ProofPreconditionVerified,
		PreconditionHashVerified:     promotion.PreconditionHashVerified,
		PreconditionReadBackVerified: promotion.PreconditionReadBackVerified,
		ProofVerified:                promotion.ProofVerified,
		ProofHashVerified:            promotion.ProofHashVerified,
		ProofReadBackVerified:        promotion.ProofReadBackVerified,
		StoreReaderVerified:          promotion.StoreReaderVerified,
		StoreVerified:                promotion.StoreVerified,
		CandidateVerified:            promotion.CandidateVerified,
		GateVerified:                 promotion.GateVerified,
		PreflightVerified:            promotion.PreflightVerified,
		BoundaryVerified:             promotion.BoundaryVerified,
		ObservationVerified:          promotion.ObservationVerified,
		ReceiverVerified:             promotion.ReceiverVerified,
		IntentVerified:               promotion.IntentVerified,
		FinalGateVerified:            promotion.FinalGateVerified,
		SealVerified:                 promotion.SealVerified,
		PermitVerified:               promotion.PermitVerified,
		AuthorityVerified:            promotion.AuthorityVerified,
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
		SourceSchema:                 promotion.Schema,
		SourceStatus:                 promotion.Status,
		SourceTarget:                 promotion.Target,
		SourceReport:                 promotionPath,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:       promotion.WeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:    promotion.WeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID: promotion.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:     promotion.PromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack: promotion.ReadBackHash,
		SourcePromotion:                                                      promotion.Promotion,
		SourcePromotionAction:                                                promotion.Action,
		SourcePromotionReceiptShape:                                          promotion.ReceiptShape,
		SourcePromotionKind:                                                  promotion.PromotionKind,
		SourcePromotionMode:                                                  promotion.PromotionMode,
		SourcePromotionStage:                                                 promotion.PromotionStage,
		SourcePromotionAdmissionRequired:                                     promotion.AdmissionRequired,
		SourcePromotionShadowOnly:                                            promotion.ShadowOnly,
		SourcePromotionGraftAllowed:                                          promotion.GraftAllowed,
		SourcePromotionDryRunOnly:                                            promotion.DryRunOnly,
		SourcePromotionLiveReady:                                             promotion.LiveReady,
		SourcePromotionRawDreamTextAllowed:                                   promotion.RawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                  promotion.RawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                 promotion.RawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                   promotion.JanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                   promotion.CoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                   promotion.DeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                   promotion.BodyMutationAllowed,
		SourcePromotionRollbackRequired:                                      promotion.RollbackRequired,
		SourcePromotionReadOnly:                                              promotion.ReadOnly,
		SourcePromotionReplayOnly:                                            promotion.ReplayOnly,
		SourcePromotionWriteAllowed:                                          promotion.WriteAllowed,
		SourcePromotionAdmissionAllowed:                                      promotion.AdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                  promotion.LiveAdmissionEnabled,
		SourcePromotionMutatesState:                                          promotion.MutatesState,
		SourcePromotionBodyTarget:                                            promotion.BodyTarget,
		SourcePromotionPassed:                                                promotion.Passed,
		SourcePromotionReason:                                                promotion.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:             promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:          promotion.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:    promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:             promotion.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:          promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:       promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:             promotion.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                     promotion.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                  promotion.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                          promotion.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                       promotion.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                     promotion.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                  promotion.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                      promotion.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                   promotion.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                        promotion.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                     promotion.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                           promotion.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                        promotion.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                          promotion.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                promotion.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                     promotion.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                   promotion.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                             promotion.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                             promotion.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                    promotion.BodySmokeWeighted,
		NanoDirectRunner:                                                     promotion.NanoDirectRunner,
		NanoDirectFinalGate:                                                  promotion.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                         promotion.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                              promotion.BoundaryReportFullChain,
		SourceAuthorityGranted:                                               promotion.SourceAuthorityGranted,
		AuthorityGranted:                                                     false,
		ContractsReady:                                                       false,
		WriteAllowed:                                                         false,
		AdmissionAllowed:                                                     false,
		LiveAdmissionEnabled:                                                 false,
		MutatesState:                                                         false,
		BodyTarget:                                                           "none",
		Passed:                                                               true,
		Reason:                                                               "weighted resonance shadow graft admission promotion held at disabled switch without mutation",
	}
	sw.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchCausalID(sw)
	sw.SwitchHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchHash(sw)
	sw.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReadBackHash(sw)
	sw.WeightedAdmissionResonanceGraftAdmissionSwitchID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchID(sw)
	if sw.CausalID == "" ||
		sw.SwitchHash == "" ||
		sw.ReadBackHash == "" ||
		sw.WeightedAdmissionResonanceGraftAdmissionSwitchID == "" ||
		sw.SwitchHash == sw.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission switch read-back proof failed")
	}
	raw, err := json.MarshalIndent(sw, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission switch marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission switch write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-switch] pass: resonance_graft_admission_switch_report=%s resonance_graft_admission_promotion_report=%s\n", outputPath, promotionPath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-switch-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission switch schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema {
		return fmt.Errorf("weighted admission resonance graft admission switch schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema)
	}
	if report.Status != "shadow_graft_admission_switch_disabled_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission switch status mismatch: got %q want %q", report.Status, "shadow_graft_admission_switch_disabled_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission switch target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission switch target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_switch")
	}
	if report.TargetMode != "closed_switch_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission switch target_mode mismatch: got %q want %q", report.TargetMode, "closed_switch_guard_dry_run")
	}
	if report.Action != "hold_weighted_resonance_shadow_graft_admission_promotion_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission switch action mismatch: got %q want %q", report.Action, "hold_weighted_resonance_shadow_graft_admission_promotion_dry_run")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission switch switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission switch switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission switch promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission switch receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_switch_receipt")
	}
	if report.SwitchKind != "shadow_graft_admission_switch" ||
		report.SwitchMode != "closed_promotion_switch_guard" ||
		report.SwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission switch shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_switch_ready", report.WeightedAdmissionResonanceGraftAdmissionSwitchReady},
		{"weighted_admission_resonance_graft_admission_promotion_consumed", report.WeightedAdmissionResonanceGraftAdmissionPromotionConsumed},
		{"weighted_admission_resonance_graft_admission_promotion_required", report.WeightedAdmissionResonanceGraftAdmissionPromotionRequired},
		{"next_step_blocked_without_resonance_graft_admission_switch", report.NextStepBlockedWithoutResonanceGraftAdmissionSwitch},
		{"promotion_verified", report.PromotionVerified},
		{"promotion_hash_verified", report.PromotionHashVerified},
		{"promotion_read_back_verified", report.PromotionReadBackVerified},
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
		{"source_weighted_admission_resonance_graft_admission_promotion_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady},
		{"source_promotion_admission_required", report.SourcePromotionAdmissionRequired},
		{"source_promotion_shadow_only", report.SourcePromotionShadowOnly},
		{"source_promotion_dry_run_only", report.SourcePromotionDryRunOnly},
		{"source_promotion_live_ready", report.SourcePromotionLiveReady},
		{"source_promotion_rollback_required", report.SourcePromotionRollbackRequired},
		{"source_promotion_read_only", report.SourcePromotionReadOnly},
		{"source_promotion_replay_only", report.SourcePromotionReplayOnly},
		{"source_promotion_passed", report.SourcePromotionPassed},
		{"source_weighted_admission_resonance_graft_admission_decision_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady},
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
			return fmt.Errorf("weighted admission resonance graft admission switch %s not ready", required.name)
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
		{"source_promotion_graft_allowed", report.SourcePromotionGraftAllowed},
		{"source_promotion_raw_dream_text_allowed", report.SourcePromotionRawDreamTextAllowed},
		{"source_promotion_raw_dream_text_observed", report.SourcePromotionRawDreamTextObserved},
		{"source_promotion_raw_dream_text_forwarded", report.SourcePromotionRawDreamTextForwarded},
		{"source_promotion_janus_surface_allowed", report.SourcePromotionJanusSurfaceAllowed},
		{"source_promotion_cooc_learning_allowed", report.SourcePromotionCoocLearningAllowed},
		{"source_promotion_delta_harvest_allowed", report.SourcePromotionDeltaHarvestAllowed},
		{"source_promotion_body_mutation_allowed", report.SourcePromotionBodyMutationAllowed},
		{"source_promotion_write_allowed", report.SourcePromotionWriteAllowed},
		{"source_promotion_admission_allowed", report.SourcePromotionAdmissionAllowed},
		{"source_promotion_live_admission_enabled", report.SourcePromotionLiveAdmissionEnabled},
		{"source_promotion_mutates_state", report.SourcePromotionMutatesState},
		{"source_authority_granted", report.SourceAuthorityGranted},
		{"authority_granted", report.AuthorityGranted},
		{"contracts_ready", report.ContractsReady},
		{"write_allowed", report.WriteAllowed},
		{"admission_allowed", report.AdmissionAllowed},
		{"live_admission_enabled", report.LiveAdmissionEnabled},
		{"mutates_state", report.MutatesState},
	} {
		if closed.value {
			return fmt.Errorf("weighted admission resonance graft admission switch opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_switch_id", report.WeightedAdmissionResonanceGraftAdmissionSwitchID},
		{"causal_id", report.CausalID},
		{"switch_hash", report.SwitchHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_promotion_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID},
		{"source_weighted_admission_resonance_graft_admission_promotion_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID},
		{"source_weighted_admission_resonance_graft_admission_promotion_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash},
		{"source_weighted_admission_resonance_graft_admission_promotion_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack},
		{"source_weighted_admission_resonance_graft_admission_decision_id", report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID},
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
			return fmt.Errorf("weighted admission resonance graft admission switch %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema {
		return fmt.Errorf("weighted admission resonance graft admission switch source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionPromotionSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_promotion_ready_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission switch source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_promotion_ready_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission switch source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission switch source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission switch source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission switch source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission switch source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission switch body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-switch-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission switch causal prefix mismatch")
	}
	if !strings.HasPrefix(report.SwitchHash, "weighted-resonance-graft-admission-switch-") {
		return fmt.Errorf("weighted admission resonance graft admission switch hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-switch-read-") ||
		report.SwitchHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission switch read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission switch source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission switch causal_id mismatch")
	}
	if report.SwitchHash == "" || report.SwitchHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission switch switch_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission switch read_back_hash mismatch")
	}
	if report.SwitchHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission switch read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionSwitchID == "" || report.WeightedAdmissionResonanceGraftAdmissionSwitchID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchID(report) {
		return fmt.Errorf("weighted admission resonance graft admission switch id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission switch reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport) string {
	h := hashJSON(struct {
		SourcePromotionID   string `json:"source_promotion_id"`
		SourcePromotionRead string `json:"source_promotion_read_back_hash"`
		SourceDecisionID    string `json:"source_decision_id"`
		SourceReaderID      string `json:"source_reader_id"`
		Target              string `json:"target"`
		SwitchKind          string `json:"switch_kind"`
		SwitchStage         string `json:"switch_stage"`
	}{
		SourcePromotionID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourcePromotionRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourceDecisionID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceReaderID:      sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		Target:              sw.Target,
		SwitchKind:          sw.SwitchKind,
		SwitchStage:         sw.SwitchStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-switch-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourcePromotionID   string `json:"source_promotion_id"`
		SourcePromotionHash string `json:"source_promotion_hash"`
		SourcePromotionRead string `json:"source_promotion_read_back_hash"`
		SwitchState         string `json:"switch_state"`
		SwitchAction        string `json:"switch_action"`
		Promotion           string `json:"promotion"`
		Action              string `json:"action"`
		ReceiptShape        string `json:"receipt_shape"`
		SwitchMode          string `json:"switch_mode"`
		PromotionVerified   bool   `json:"promotion_verified"`
		ReadOnly            bool   `json:"read_only"`
		ReplayOnly          bool   `json:"replay_only"`
		AdmissionRequired   bool   `json:"admission_required"`
		ShadowOnly          bool   `json:"shadow_only"`
		DryRunOnly          bool   `json:"dry_run_only"`
		GraftAllowed        bool   `json:"graft_allowed"`
		BodyMutation        bool   `json:"body_mutation_allowed"`
		LiveAdmission       bool   `json:"live_admission_enabled"`
	}{
		CausalID:            sw.CausalID,
		SourcePromotionID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourcePromotionHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourcePromotionRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SwitchState:         sw.SwitchState,
		SwitchAction:        sw.SwitchAction,
		Promotion:           sw.Promotion,
		Action:              sw.Action,
		ReceiptShape:        sw.ReceiptShape,
		SwitchMode:          sw.SwitchMode,
		PromotionVerified:   sw.PromotionVerified,
		ReadOnly:            sw.ReadOnly,
		ReplayOnly:          sw.ReplayOnly,
		AdmissionRequired:   sw.AdmissionRequired,
		ShadowOnly:          sw.ShadowOnly,
		DryRunOnly:          sw.DryRunOnly,
		GraftAllowed:        sw.GraftAllowed,
		BodyMutation:        sw.BodyMutationAllowed,
		LiveAdmission:       sw.LiveAdmissionEnabled,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-switch-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport) string {
	h := hashJSON(struct {
		SwitchHash          string `json:"switch_hash"`
		SourcePromotionID   string `json:"source_promotion_id"`
		SourcePromotionRead string `json:"source_promotion_read_back_hash"`
		SwitchKind          string `json:"switch_kind"`
		SwitchReady         bool   `json:"switch_ready"`
		PromotionConsumed   bool   `json:"promotion_consumed"`
		LiveReady           bool   `json:"live_ready"`
		BodyMutation        bool   `json:"body_mutation"`
		LiveAdmission       bool   `json:"live_admission"`
		WriteAllowed        bool   `json:"write_allowed"`
		AdmissionAllowed    bool   `json:"admission_allowed"`
	}{
		SwitchHash:          sw.SwitchHash,
		SourcePromotionID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourcePromotionRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SwitchKind:          sw.SwitchKind,
		SwitchReady:         sw.WeightedAdmissionResonanceGraftAdmissionSwitchReady,
		PromotionConsumed:   sw.WeightedAdmissionResonanceGraftAdmissionPromotionConsumed,
		LiveReady:           sw.LiveReady,
		BodyMutation:        sw.BodyMutationAllowed,
		LiveAdmission:       sw.LiveAdmissionEnabled,
		WriteAllowed:        sw.WriteAllowed,
		AdmissionAllowed:    sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-switch-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		SwitchState             string `json:"switch_state"`
		SwitchAction            string `json:"switch_action"`
		Promotion               string `json:"promotion"`
		SourceReport            string `json:"source_report"`
		SourcePromotionID       string `json:"source_promotion_id"`
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
		SwitchHash              string `json:"switch_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		SwitchKind              string `json:"switch_kind"`
		SwitchMode              string `json:"switch_mode"`
		SwitchStage             string `json:"switch_stage"`
		PromotionVerified       bool   `json:"promotion_verified"`
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
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_switch"`
		SourcePromotionReady    bool   `json:"source_promotion_ready"`
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
		Schema:                  sw.Schema,
		Status:                  sw.Status,
		Action:                  sw.Action,
		SwitchState:             sw.SwitchState,
		SwitchAction:            sw.SwitchAction,
		Promotion:               sw.Promotion,
		SourceReport:            sw.SourceReport,
		SourcePromotionID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceDecisionID:        sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceProofID:           sw.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceReaderID:          sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceStoreID:           sw.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceCandidateID:       sw.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceGateID:            sw.SourceWeightedAdmissionResonanceGraftGateID,
		SourcePreflightID:       sw.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceBoundaryID:        sw.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceObservationID:     sw.SourceWeightedAdmissionResonanceObservationID,
		SourceReceiverID:        sw.SourceWeightedAdmissionResonanceReceiverID,
		CausalID:                sw.CausalID,
		SwitchHash:              sw.SwitchHash,
		ReadBackHash:            sw.ReadBackHash,
		Ready:                   sw.WeightedAdmissionResonanceGraftAdmissionSwitchReady,
		ReceiptShape:            sw.ReceiptShape,
		SwitchKind:              sw.SwitchKind,
		SwitchMode:              sw.SwitchMode,
		SwitchStage:             sw.SwitchStage,
		PromotionVerified:       sw.PromotionVerified,
		AdmissionRequired:       sw.AdmissionRequired,
		ShadowOnly:              sw.ShadowOnly,
		GraftAllowed:            sw.GraftAllowed,
		DryRunOnly:              sw.DryRunOnly,
		RawDreamTextAllowed:     sw.RawDreamTextAllowed,
		JanusSurfaceAllowed:     sw.JanusSurfaceAllowed,
		CoocLearningAllowed:     sw.CoocLearningAllowed,
		DeltaHarvestAllowed:     sw.DeltaHarvestAllowed,
		BodyMutationAllowed:     sw.BodyMutationAllowed,
		RollbackRequired:        sw.RollbackRequired,
		ReadOnly:                sw.ReadOnly,
		ReplayOnly:              sw.ReplayOnly,
		LiveReady:               sw.LiveReady,
		ContractsReady:          sw.ContractsReady,
		BodyTarget:              sw.BodyTarget,
		WriteAllowed:            sw.WriteAllowed,
		AdmissionAllowed:        sw.AdmissionAllowed,
		LiveAdmissionEnabled:    sw.LiveAdmissionEnabled,
		MutatesState:            sw.MutatesState,
		NextStepBlockedWithout:  sw.NextStepBlockedWithoutResonanceGraftAdmissionSwitch,
		SourcePromotionReady:    sw.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceDecisionReady:     sw.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourcePreconditionReady: sw.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceProofReady:        sw.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceReaderReady:       sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceStoreReady:        sw.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceCandidateReady:    sw.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceGateReady:         sw.SourceWeightedAdmissionResonanceGraftGateReady,
		SourcePreflightReady:    sw.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceBoundaryReady:     sw.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceObservationReady:  sw.SourceWeightedAdmissionResonanceObservationReady,
		SourceReceiverReady:     sw.SourceWeightedAdmissionResonanceReceiverReady,
		SourceIntentReady:       sw.SourceWeightedAdmissionResonanceIntentReady,
		SourceFinalGateReady:    sw.SourceWeightedAdmissionFinalGateReady,
		SourceSealReady:         sw.SourceWeightedAdmissionSealReady,
		SourcePermitReady:       sw.SourceWeightedAdmissionPermitReady,
		SourceAuthorityUsed:     sw.SourceWeightedAdmissionAuthorityConsumed,
		SourceAuthorityNeeded:   sw.SourceWeightedAdmissionAuthorityRequired,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-switch-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission switch path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission switch not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission switch not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission switch JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission switch decode failed: %w", err)
	}
	return report, root, nil
}
