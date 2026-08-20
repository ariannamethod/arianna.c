package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema = "arianna.live_route_weighted_admission_resonance_graft_admission_writer_preflight.v1"

type admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport struct {
	Schema                                                               string `json:"schema"`
	Status                                                               string `json:"status"`
	Target                                                               string `json:"target"`
	TargetKind                                                           string `json:"target_kind"`
	TargetMode                                                           string `json:"target_mode"`
	Action                                                               string `json:"action"`
	WriterState                                                          string `json:"writer_state"`
	WriterAction                                                         string `json:"writer_action"`
	RollbackState                                                        string `json:"rollback_state"`
	RollbackAction                                                       string `json:"rollback_action"`
	StageState                                                           string `json:"stage_state"`
	StageAction                                                          string `json:"stage_action"`
	EnableState                                                          string `json:"enable_state"`
	EnableAction                                                         string `json:"enable_action"`
	SwitchState                                                          string `json:"switch_state"`
	SwitchAction                                                         string `json:"switch_action"`
	Promotion                                                            string `json:"promotion"`
	WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady         bool   `json:"weighted_admission_resonance_graft_admission_writer_preflight_ready"`
	WeightedAdmissionResonanceGraftAdmissionLiveStageConsumed            bool   `json:"weighted_admission_resonance_graft_admission_live_stage_consumed"`
	WeightedAdmissionResonanceGraftAdmissionLiveStageRequired            bool   `json:"weighted_admission_resonance_graft_admission_live_stage_required"`
	NextStepBlockedWithoutResonanceGraftAdmissionWriterPreflight         bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_preflight"`
	WeightedAdmissionResonanceGraftAdmissionWriterPreflightID            string `json:"weighted_admission_resonance_graft_admission_writer_preflight_id"`
	ReceiptShape                                                         string `json:"receipt_shape"`
	WriterPreflightKind                                                  string `json:"writer_preflight_kind"`
	WriterPreflightMode                                                  string `json:"writer_preflight_mode"`
	WriterPreflightStage                                                 string `json:"writer_preflight_stage"`
	CausalID                                                             string `json:"causal_id"`
	WriterPreflightHash                                                  string `json:"writer_preflight_hash"`
	ReadBackHash                                                         string `json:"read_back_hash"`
	LiveStageVerified                                                    bool   `json:"live_stage_verified"`
	LiveStageHashVerified                                                bool   `json:"live_stage_hash_verified"`
	LiveStageReadBackVerified                                            bool   `json:"live_stage_read_back_verified"`
	EnableGateVerified                                                   bool   `json:"enable_gate_verified"`
	EnableGateHashVerified                                               bool   `json:"enable_gate_hash_verified"`
	EnableGateReadBackVerified                                           bool   `json:"enable_gate_read_back_verified"`
	SwitchVerified                                                       bool   `json:"switch_verified"`
	SwitchHashVerified                                                   bool   `json:"switch_hash_verified"`
	SwitchReadBackVerified                                               bool   `json:"switch_read_back_verified"`
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
	RequiresWriter                                                       bool   `json:"requires_writer"`
	WriterReady                                                          bool   `json:"writer_ready"`
	RollbackRequired                                                     bool   `json:"rollback_required"`
	RequiresRollback                                                     bool   `json:"requires_rollback"`
	RollbackReady                                                        bool   `json:"rollback_ready"`
	ReadOnly                                                             bool   `json:"read_only"`
	ReplayOnly                                                           bool   `json:"replay_only"`
	SourceSchema                                                         string `json:"source_schema"`
	SourceStatus                                                         string `json:"source_status"`
	SourceTarget                                                         string `json:"source_target"`
	SourceReport                                                         string `json:"source_report"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID            string `json:"source_weighted_admission_resonance_graft_admission_live_stage_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady         bool   `json:"source_weighted_admission_resonance_graft_admission_live_stage_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID      string `json:"source_weighted_admission_resonance_graft_admission_live_stage_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash          string `json:"source_weighted_admission_resonance_graft_admission_live_stage_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack      string `json:"source_weighted_admission_resonance_graft_admission_live_stage_read_back_hash"`
	SourceStageState                                                     string `json:"source_stage_state"`
	SourceStageAction                                                    string `json:"source_stage_action"`
	SourceLiveStageReceiptShape                                          string `json:"source_live_stage_receipt_shape"`
	SourceLiveStageKind                                                  string `json:"source_live_stage_kind"`
	SourceLiveStageMode                                                  string `json:"source_live_stage_mode"`
	SourceLiveStageStage                                                 string `json:"source_live_stage_stage"`
	SourceLiveStageAdmissionRequired                                     bool   `json:"source_live_stage_admission_required"`
	SourceLiveStageShadowOnly                                            bool   `json:"source_live_stage_shadow_only"`
	SourceLiveStageGraftAllowed                                          bool   `json:"source_live_stage_graft_allowed"`
	SourceLiveStageDryRunOnly                                            bool   `json:"source_live_stage_dry_run_only"`
	SourceLiveStageLiveReady                                             bool   `json:"source_live_stage_live_ready"`
	SourceLiveStageRawDreamTextAllowed                                   bool   `json:"source_live_stage_raw_dream_text_allowed"`
	SourceLiveStageRawDreamTextObserved                                  bool   `json:"source_live_stage_raw_dream_text_observed"`
	SourceLiveStageRawDreamTextForwarded                                 bool   `json:"source_live_stage_raw_dream_text_forwarded"`
	SourceLiveStageJanusSurfaceAllowed                                   bool   `json:"source_live_stage_janus_surface_allowed"`
	SourceLiveStageCoocLearningAllowed                                   bool   `json:"source_live_stage_cooc_learning_allowed"`
	SourceLiveStageDeltaHarvestAllowed                                   bool   `json:"source_live_stage_delta_harvest_allowed"`
	SourceLiveStageBodyMutationAllowed                                   bool   `json:"source_live_stage_body_mutation_allowed"`
	SourceLiveStageRequiresWriter                                        bool   `json:"source_live_stage_requires_writer"`
	SourceLiveStageWriterReady                                           bool   `json:"source_live_stage_writer_ready"`
	SourceLiveStageRollbackRequired                                      bool   `json:"source_live_stage_rollback_required"`
	SourceLiveStageRequiresRollback                                      bool   `json:"source_live_stage_requires_rollback"`
	SourceLiveStageRollbackReady                                         bool   `json:"source_live_stage_rollback_ready"`
	SourceLiveStageReadOnly                                              bool   `json:"source_live_stage_read_only"`
	SourceLiveStageReplayOnly                                            bool   `json:"source_live_stage_replay_only"`
	SourceLiveStageWriteAllowed                                          bool   `json:"source_live_stage_write_allowed"`
	SourceLiveStageAdmissionAllowed                                      bool   `json:"source_live_stage_admission_allowed"`
	SourceLiveStageLiveAdmissionEnabled                                  bool   `json:"source_live_stage_live_admission_enabled"`
	SourceLiveStageMutatesState                                          bool   `json:"source_live_stage_mutates_state"`
	SourceLiveStageBodyTarget                                            string `json:"source_live_stage_body_target"`
	SourceLiveStagePassed                                                bool   `json:"source_live_stage_passed"`
	SourceLiveStageReason                                                string `json:"source_live_stage_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID           string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady        bool   `json:"source_weighted_admission_resonance_graft_admission_enable_gate_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID     string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash         string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack     string `json:"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash"`
	SourceEnableState                                                    string `json:"source_enable_state"`
	SourceEnableAction                                                   string `json:"source_enable_action"`
	SourceEnableGateReceiptShape                                         string `json:"source_enable_gate_receipt_shape"`
	SourceEnableGateKind                                                 string `json:"source_enable_gate_kind"`
	SourceEnableGateMode                                                 string `json:"source_enable_gate_mode"`
	SourceEnableGateStage                                                string `json:"source_enable_gate_stage"`
	SourceEnableGateAdmissionRequired                                    bool   `json:"source_enable_gate_admission_required"`
	SourceEnableGateShadowOnly                                           bool   `json:"source_enable_gate_shadow_only"`
	SourceEnableGateGraftAllowed                                         bool   `json:"source_enable_gate_graft_allowed"`
	SourceEnableGateDryRunOnly                                           bool   `json:"source_enable_gate_dry_run_only"`
	SourceEnableGateLiveReady                                            bool   `json:"source_enable_gate_live_ready"`
	SourceEnableGateRawDreamTextAllowed                                  bool   `json:"source_enable_gate_raw_dream_text_allowed"`
	SourceEnableGateRawDreamTextObserved                                 bool   `json:"source_enable_gate_raw_dream_text_observed"`
	SourceEnableGateRawDreamTextForwarded                                bool   `json:"source_enable_gate_raw_dream_text_forwarded"`
	SourceEnableGateJanusSurfaceAllowed                                  bool   `json:"source_enable_gate_janus_surface_allowed"`
	SourceEnableGateCoocLearningAllowed                                  bool   `json:"source_enable_gate_cooc_learning_allowed"`
	SourceEnableGateDeltaHarvestAllowed                                  bool   `json:"source_enable_gate_delta_harvest_allowed"`
	SourceEnableGateBodyMutationAllowed                                  bool   `json:"source_enable_gate_body_mutation_allowed"`
	SourceEnableGateRollbackRequired                                     bool   `json:"source_enable_gate_rollback_required"`
	SourceEnableGateReadOnly                                             bool   `json:"source_enable_gate_read_only"`
	SourceEnableGateReplayOnly                                           bool   `json:"source_enable_gate_replay_only"`
	SourceEnableGateWriteAllowed                                         bool   `json:"source_enable_gate_write_allowed"`
	SourceEnableGateAdmissionAllowed                                     bool   `json:"source_enable_gate_admission_allowed"`
	SourceEnableGateLiveAdmissionEnabled                                 bool   `json:"source_enable_gate_live_admission_enabled"`
	SourceEnableGateMutatesState                                         bool   `json:"source_enable_gate_mutates_state"`
	SourceEnableGateBodyTarget                                           string `json:"source_enable_gate_body_target"`
	SourceEnableGatePassed                                               bool   `json:"source_enable_gate_passed"`
	SourceEnableGateReason                                               string `json:"source_enable_gate_reason"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchID               string `json:"source_weighted_admission_resonance_graft_admission_switch_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady            bool   `json:"source_weighted_admission_resonance_graft_admission_switch_ready"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID         string `json:"source_weighted_admission_resonance_graft_admission_switch_causal_id"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash             string `json:"source_weighted_admission_resonance_graft_admission_switch_hash"`
	SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack         string `json:"source_weighted_admission_resonance_graft_admission_switch_read_back_hash"`
	SourceSwitchState                                                    string `json:"source_switch_state"`
	SourceSwitchAction                                                   string `json:"source_switch_action"`
	SourceSwitchReceiptShape                                             string `json:"source_switch_receipt_shape"`
	SourceSwitchKind                                                     string `json:"source_switch_kind"`
	SourceSwitchMode                                                     string `json:"source_switch_mode"`
	SourceSwitchStage                                                    string `json:"source_switch_stage"`
	SourceSwitchAdmissionRequired                                        bool   `json:"source_switch_admission_required"`
	SourceSwitchShadowOnly                                               bool   `json:"source_switch_shadow_only"`
	SourceSwitchGraftAllowed                                             bool   `json:"source_switch_graft_allowed"`
	SourceSwitchDryRunOnly                                               bool   `json:"source_switch_dry_run_only"`
	SourceSwitchLiveReady                                                bool   `json:"source_switch_live_ready"`
	SourceSwitchRawDreamTextAllowed                                      bool   `json:"source_switch_raw_dream_text_allowed"`
	SourceSwitchRawDreamTextObserved                                     bool   `json:"source_switch_raw_dream_text_observed"`
	SourceSwitchRawDreamTextForwarded                                    bool   `json:"source_switch_raw_dream_text_forwarded"`
	SourceSwitchJanusSurfaceAllowed                                      bool   `json:"source_switch_janus_surface_allowed"`
	SourceSwitchCoocLearningAllowed                                      bool   `json:"source_switch_cooc_learning_allowed"`
	SourceSwitchDeltaHarvestAllowed                                      bool   `json:"source_switch_delta_harvest_allowed"`
	SourceSwitchBodyMutationAllowed                                      bool   `json:"source_switch_body_mutation_allowed"`
	SourceSwitchRollbackRequired                                         bool   `json:"source_switch_rollback_required"`
	SourceSwitchReadOnly                                                 bool   `json:"source_switch_read_only"`
	SourceSwitchReplayOnly                                               bool   `json:"source_switch_replay_only"`
	SourceSwitchWriteAllowed                                             bool   `json:"source_switch_write_allowed"`
	SourceSwitchAdmissionAllowed                                         bool   `json:"source_switch_admission_allowed"`
	SourceSwitchLiveAdmissionEnabled                                     bool   `json:"source_switch_live_admission_enabled"`
	SourceSwitchMutatesState                                             bool   `json:"source_switch_mutates_state"`
	SourceSwitchBodyTarget                                               string `json:"source_switch_body_target"`
	SourceSwitchPassed                                                   bool   `json:"source_switch_passed"`
	SourceSwitchReason                                                   string `json:"source_switch_reason"`
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

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT")
	}
	liveStagePath := args[0]
	outputPath := args[1]
	if strings.TrimSpace(outputPath) == "" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight output path missing")
	}
	sourceStage, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportForAssert(liveStagePath)
	if err != nil {
		return err
	}
	if err := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReportError(sourceStage, root); err != nil {
		return err
	}
	stage := admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport{
		Schema:               admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema,
		Status:               "shadow_graft_admission_writer_preflight_blocked_dry_run",
		Target:               "live_route_admission_next_step",
		TargetKind:           "weighted_internal_world_shadow_graft_admission_writer_preflight",
		TargetMode:           "closed_writer_preflight_guard_dry_run",
		Action:               "block_weighted_resonance_shadow_graft_admission_live_stage_blocked_dry_run",
		WriterState:          "blocked",
		WriterAction:         "reject_blocked_live_stage",
		RollbackState:        "blocked",
		RollbackAction:       "reject_blocked_live_stage",
		StageState:           sourceStage.StageState,
		StageAction:          sourceStage.StageAction,
		EnableState:          sourceStage.EnableState,
		EnableAction:         sourceStage.EnableAction,
		SwitchState:          sourceStage.SwitchState,
		SwitchAction:         sourceStage.SwitchAction,
		Promotion:            sourceStage.Promotion,
		ReceiptShape:         "weighted_resonance_shadow_graft_admission_writer_preflight_receipt",
		WriterPreflightKind:  "shadow_graft_admission_writer_preflight",
		WriterPreflightMode:  "closed_live_stage_writer_preflight_guard",
		WriterPreflightStage: "pre_writer_inventory_graft_admission_writer_preflight",
		WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady: true,
		WeightedAdmissionResonanceGraftAdmissionLiveStageConsumed:    true,
		WeightedAdmissionResonanceGraftAdmissionLiveStageRequired:    true,
		NextStepBlockedWithoutResonanceGraftAdmissionWriterPreflight: true,
		LiveStageVerified:            true,
		LiveStageHashVerified:        true,
		LiveStageReadBackVerified:    true,
		EnableGateVerified:           sourceStage.EnableGateVerified,
		EnableGateHashVerified:       sourceStage.EnableGateHashVerified,
		EnableGateReadBackVerified:   sourceStage.EnableGateReadBackVerified,
		SwitchVerified:               sourceStage.SwitchVerified,
		SwitchHashVerified:           sourceStage.SwitchHashVerified,
		SwitchReadBackVerified:       sourceStage.SwitchReadBackVerified,
		PromotionVerified:            sourceStage.PromotionVerified,
		PromotionHashVerified:        sourceStage.PromotionHashVerified,
		PromotionReadBackVerified:    sourceStage.PromotionReadBackVerified,
		DecisionVerified:             sourceStage.DecisionVerified,
		DecisionHashVerified:         sourceStage.DecisionHashVerified,
		DecisionReadBackVerified:     sourceStage.DecisionReadBackVerified,
		ProofPreconditionVerified:    sourceStage.ProofPreconditionVerified,
		PreconditionHashVerified:     sourceStage.PreconditionHashVerified,
		PreconditionReadBackVerified: sourceStage.PreconditionReadBackVerified,
		ProofVerified:                sourceStage.ProofVerified,
		ProofHashVerified:            sourceStage.ProofHashVerified,
		ProofReadBackVerified:        sourceStage.ProofReadBackVerified,
		StoreReaderVerified:          sourceStage.StoreReaderVerified,
		StoreVerified:                sourceStage.StoreVerified,
		CandidateVerified:            sourceStage.CandidateVerified,
		GateVerified:                 sourceStage.GateVerified,
		PreflightVerified:            sourceStage.PreflightVerified,
		BoundaryVerified:             sourceStage.BoundaryVerified,
		ObservationVerified:          sourceStage.ObservationVerified,
		ReceiverVerified:             sourceStage.ReceiverVerified,
		IntentVerified:               sourceStage.IntentVerified,
		FinalGateVerified:            sourceStage.FinalGateVerified,
		SealVerified:                 sourceStage.SealVerified,
		PermitVerified:               sourceStage.PermitVerified,
		AuthorityVerified:            sourceStage.AuthorityVerified,
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
		RequiresWriter:               true,
		WriterReady:                  false,
		RollbackRequired:             true,
		RequiresRollback:             true,
		RollbackReady:                false,
		ReadOnly:                     true,
		ReplayOnly:                   true,
		SourceSchema:                 sourceStage.Schema,
		SourceStatus:                 sourceStage.Status,
		SourceTarget:                 sourceStage.Target,
		SourceReport:                 liveStagePath,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID:       sourceStage.WeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady:    sourceStage.WeightedAdmissionResonanceGraftAdmissionLiveStageReady,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID: sourceStage.CausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash:     sourceStage.LiveStageHash,
		SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack: sourceStage.ReadBackHash,
		SourceStageState:                                                     sourceStage.StageState,
		SourceStageAction:                                                    sourceStage.StageAction,
		SourceLiveStageReceiptShape:                                          sourceStage.ReceiptShape,
		SourceLiveStageKind:                                                  sourceStage.LiveStageKind,
		SourceLiveStageMode:                                                  sourceStage.LiveStageMode,
		SourceLiveStageStage:                                                 sourceStage.LiveStageStage,
		SourceLiveStageAdmissionRequired:                                     sourceStage.AdmissionRequired,
		SourceLiveStageShadowOnly:                                            sourceStage.ShadowOnly,
		SourceLiveStageGraftAllowed:                                          sourceStage.GraftAllowed,
		SourceLiveStageDryRunOnly:                                            sourceStage.DryRunOnly,
		SourceLiveStageLiveReady:                                             sourceStage.LiveReady,
		SourceLiveStageRawDreamTextAllowed:                                   sourceStage.RawDreamTextAllowed,
		SourceLiveStageRawDreamTextObserved:                                  sourceStage.RawDreamTextObserved,
		SourceLiveStageRawDreamTextForwarded:                                 sourceStage.RawDreamTextForwarded,
		SourceLiveStageJanusSurfaceAllowed:                                   sourceStage.JanusSurfaceAllowed,
		SourceLiveStageCoocLearningAllowed:                                   sourceStage.CoocLearningAllowed,
		SourceLiveStageDeltaHarvestAllowed:                                   sourceStage.DeltaHarvestAllowed,
		SourceLiveStageBodyMutationAllowed:                                   sourceStage.BodyMutationAllowed,
		SourceLiveStageRequiresWriter:                                        sourceStage.RequiresWriter,
		SourceLiveStageWriterReady:                                           sourceStage.WriterReady,
		SourceLiveStageRollbackRequired:                                      sourceStage.RollbackRequired,
		SourceLiveStageRequiresRollback:                                      sourceStage.RequiresRollback,
		SourceLiveStageRollbackReady:                                         sourceStage.RollbackReady,
		SourceLiveStageReadOnly:                                              sourceStage.ReadOnly,
		SourceLiveStageReplayOnly:                                            sourceStage.ReplayOnly,
		SourceLiveStageWriteAllowed:                                          sourceStage.WriteAllowed,
		SourceLiveStageAdmissionAllowed:                                      sourceStage.AdmissionAllowed,
		SourceLiveStageLiveAdmissionEnabled:                                  sourceStage.LiveAdmissionEnabled,
		SourceLiveStageMutatesState:                                          sourceStage.MutatesState,
		SourceLiveStageBodyTarget:                                            sourceStage.BodyTarget,
		SourceLiveStagePassed:                                                sourceStage.Passed,
		SourceLiveStageReason:                                                sourceStage.Reason,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID:           sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady:        sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID:     sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash:         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack:     sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceEnableState:                                                    sourceStage.SourceEnableState,
		SourceEnableAction:                                                   sourceStage.SourceEnableAction,
		SourceEnableGateReceiptShape:                                         sourceStage.SourceEnableGateReceiptShape,
		SourceEnableGateKind:                                                 sourceStage.SourceEnableGateKind,
		SourceEnableGateMode:                                                 sourceStage.SourceEnableGateMode,
		SourceEnableGateStage:                                                sourceStage.SourceEnableGateStage,
		SourceEnableGateAdmissionRequired:                                    sourceStage.SourceEnableGateAdmissionRequired,
		SourceEnableGateShadowOnly:                                           sourceStage.SourceEnableGateShadowOnly,
		SourceEnableGateGraftAllowed:                                         sourceStage.SourceEnableGateGraftAllowed,
		SourceEnableGateDryRunOnly:                                           sourceStage.SourceEnableGateDryRunOnly,
		SourceEnableGateLiveReady:                                            sourceStage.SourceEnableGateLiveReady,
		SourceEnableGateRawDreamTextAllowed:                                  sourceStage.SourceEnableGateRawDreamTextAllowed,
		SourceEnableGateRawDreamTextObserved:                                 sourceStage.SourceEnableGateRawDreamTextObserved,
		SourceEnableGateRawDreamTextForwarded:                                sourceStage.SourceEnableGateRawDreamTextForwarded,
		SourceEnableGateJanusSurfaceAllowed:                                  sourceStage.SourceEnableGateJanusSurfaceAllowed,
		SourceEnableGateCoocLearningAllowed:                                  sourceStage.SourceEnableGateCoocLearningAllowed,
		SourceEnableGateDeltaHarvestAllowed:                                  sourceStage.SourceEnableGateDeltaHarvestAllowed,
		SourceEnableGateBodyMutationAllowed:                                  sourceStage.SourceEnableGateBodyMutationAllowed,
		SourceEnableGateRollbackRequired:                                     sourceStage.SourceEnableGateRollbackRequired,
		SourceEnableGateReadOnly:                                             sourceStage.SourceEnableGateReadOnly,
		SourceEnableGateReplayOnly:                                           sourceStage.SourceEnableGateReplayOnly,
		SourceEnableGateWriteAllowed:                                         sourceStage.SourceEnableGateWriteAllowed,
		SourceEnableGateAdmissionAllowed:                                     sourceStage.SourceEnableGateAdmissionAllowed,
		SourceEnableGateLiveAdmissionEnabled:                                 sourceStage.SourceEnableGateLiveAdmissionEnabled,
		SourceEnableGateMutatesState:                                         sourceStage.SourceEnableGateMutatesState,
		SourceEnableGateBodyTarget:                                           sourceStage.SourceEnableGateBodyTarget,
		SourceEnableGatePassed:                                               sourceStage.SourceEnableGatePassed,
		SourceEnableGateReason:                                               sourceStage.SourceEnableGateReason,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchID:               sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady:            sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID:         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash:             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack:         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
		SourceSwitchState:                                                    sourceStage.SourceSwitchState,
		SourceSwitchAction:                                                   sourceStage.SourceSwitchAction,
		SourceSwitchReceiptShape:                                             sourceStage.SourceSwitchReceiptShape,
		SourceSwitchKind:                                                     sourceStage.SourceSwitchKind,
		SourceSwitchMode:                                                     sourceStage.SourceSwitchMode,
		SourceSwitchStage:                                                    sourceStage.SourceSwitchStage,
		SourceSwitchAdmissionRequired:                                        sourceStage.SourceSwitchAdmissionRequired,
		SourceSwitchShadowOnly:                                               sourceStage.SourceSwitchShadowOnly,
		SourceSwitchGraftAllowed:                                             sourceStage.SourceSwitchGraftAllowed,
		SourceSwitchDryRunOnly:                                               sourceStage.SourceSwitchDryRunOnly,
		SourceSwitchLiveReady:                                                sourceStage.SourceSwitchLiveReady,
		SourceSwitchRawDreamTextAllowed:                                      sourceStage.SourceSwitchRawDreamTextAllowed,
		SourceSwitchRawDreamTextObserved:                                     sourceStage.SourceSwitchRawDreamTextObserved,
		SourceSwitchRawDreamTextForwarded:                                    sourceStage.SourceSwitchRawDreamTextForwarded,
		SourceSwitchJanusSurfaceAllowed:                                      sourceStage.SourceSwitchJanusSurfaceAllowed,
		SourceSwitchCoocLearningAllowed:                                      sourceStage.SourceSwitchCoocLearningAllowed,
		SourceSwitchDeltaHarvestAllowed:                                      sourceStage.SourceSwitchDeltaHarvestAllowed,
		SourceSwitchBodyMutationAllowed:                                      sourceStage.SourceSwitchBodyMutationAllowed,
		SourceSwitchRollbackRequired:                                         sourceStage.SourceSwitchRollbackRequired,
		SourceSwitchReadOnly:                                                 sourceStage.SourceSwitchReadOnly,
		SourceSwitchReplayOnly:                                               sourceStage.SourceSwitchReplayOnly,
		SourceSwitchWriteAllowed:                                             sourceStage.SourceSwitchWriteAllowed,
		SourceSwitchAdmissionAllowed:                                         sourceStage.SourceSwitchAdmissionAllowed,
		SourceSwitchLiveAdmissionEnabled:                                     sourceStage.SourceSwitchLiveAdmissionEnabled,
		SourceSwitchMutatesState:                                             sourceStage.SourceSwitchMutatesState,
		SourceSwitchBodyTarget:                                               sourceStage.SourceSwitchBodyTarget,
		SourceSwitchPassed:                                                   sourceStage.SourceSwitchPassed,
		SourceSwitchReason:                                                   sourceStage.SourceSwitchReason,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionID:            sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady:         sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID:      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash:          sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash,
		SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack:      sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack,
		SourcePromotion:                                                      sourceStage.SourcePromotion,
		SourcePromotionAction:                                                sourceStage.SourcePromotionAction,
		SourcePromotionReceiptShape:                                          sourceStage.SourcePromotionReceiptShape,
		SourcePromotionKind:                                                  sourceStage.SourcePromotionKind,
		SourcePromotionMode:                                                  sourceStage.SourcePromotionMode,
		SourcePromotionStage:                                                 sourceStage.SourcePromotionStage,
		SourcePromotionAdmissionRequired:                                     sourceStage.SourcePromotionAdmissionRequired,
		SourcePromotionShadowOnly:                                            sourceStage.SourcePromotionShadowOnly,
		SourcePromotionGraftAllowed:                                          sourceStage.SourcePromotionGraftAllowed,
		SourcePromotionDryRunOnly:                                            sourceStage.SourcePromotionDryRunOnly,
		SourcePromotionLiveReady:                                             sourceStage.SourcePromotionLiveReady,
		SourcePromotionRawDreamTextAllowed:                                   sourceStage.SourcePromotionRawDreamTextAllowed,
		SourcePromotionRawDreamTextObserved:                                  sourceStage.SourcePromotionRawDreamTextObserved,
		SourcePromotionRawDreamTextForwarded:                                 sourceStage.SourcePromotionRawDreamTextForwarded,
		SourcePromotionJanusSurfaceAllowed:                                   sourceStage.SourcePromotionJanusSurfaceAllowed,
		SourcePromotionCoocLearningAllowed:                                   sourceStage.SourcePromotionCoocLearningAllowed,
		SourcePromotionDeltaHarvestAllowed:                                   sourceStage.SourcePromotionDeltaHarvestAllowed,
		SourcePromotionBodyMutationAllowed:                                   sourceStage.SourcePromotionBodyMutationAllowed,
		SourcePromotionRollbackRequired:                                      sourceStage.SourcePromotionRollbackRequired,
		SourcePromotionReadOnly:                                              sourceStage.SourcePromotionReadOnly,
		SourcePromotionReplayOnly:                                            sourceStage.SourcePromotionReplayOnly,
		SourcePromotionWriteAllowed:                                          sourceStage.SourcePromotionWriteAllowed,
		SourcePromotionAdmissionAllowed:                                      sourceStage.SourcePromotionAdmissionAllowed,
		SourcePromotionLiveAdmissionEnabled:                                  sourceStage.SourcePromotionLiveAdmissionEnabled,
		SourcePromotionMutatesState:                                          sourceStage.SourcePromotionMutatesState,
		SourcePromotionBodyTarget:                                            sourceStage.SourcePromotionBodyTarget,
		SourcePromotionPassed:                                                sourceStage.SourcePromotionPassed,
		SourcePromotionReason:                                                sourceStage.SourcePromotionReason,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionID:             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID,
		SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady:          sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionDecisionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID:    sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady: sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionReady,
		SourceWeightedAdmissionResonanceGraftAdmissionProofID:                sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionProofID,
		SourceWeightedAdmissionResonanceGraftAdmissionProofReady:             sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionProofReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID:          sourceStage.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady:       sourceStage.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderReady,
		SourceWeightedAdmissionResonanceGraftCandidateStoreID:                sourceStage.SourceWeightedAdmissionResonanceGraftCandidateStoreID,
		SourceWeightedAdmissionResonanceGraftCandidateStoreReady:             sourceStage.SourceWeightedAdmissionResonanceGraftCandidateStoreReady,
		SourceWeightedAdmissionResonanceGraftCandidateID:                     sourceStage.SourceWeightedAdmissionResonanceGraftCandidateID,
		SourceWeightedAdmissionResonanceGraftCandidateReady:                  sourceStage.SourceWeightedAdmissionResonanceGraftCandidateReady,
		SourceWeightedAdmissionResonanceGraftGateID:                          sourceStage.SourceWeightedAdmissionResonanceGraftGateID,
		SourceWeightedAdmissionResonanceGraftGateReady:                       sourceStage.SourceWeightedAdmissionResonanceGraftGateReady,
		SourceWeightedAdmissionResonanceGraftPreflightID:                     sourceStage.SourceWeightedAdmissionResonanceGraftPreflightID,
		SourceWeightedAdmissionResonanceGraftPreflightReady:                  sourceStage.SourceWeightedAdmissionResonanceGraftPreflightReady,
		SourceWeightedAdmissionResonanceGraftBoundaryID:                      sourceStage.SourceWeightedAdmissionResonanceGraftBoundaryID,
		SourceWeightedAdmissionResonanceGraftBoundaryReady:                   sourceStage.SourceWeightedAdmissionResonanceGraftBoundaryReady,
		SourceWeightedAdmissionResonanceObservationID:                        sourceStage.SourceWeightedAdmissionResonanceObservationID,
		SourceWeightedAdmissionResonanceObservationReady:                     sourceStage.SourceWeightedAdmissionResonanceObservationReady,
		SourceWeightedAdmissionResonanceReceiverID:                           sourceStage.SourceWeightedAdmissionResonanceReceiverID,
		SourceWeightedAdmissionResonanceReceiverReady:                        sourceStage.SourceWeightedAdmissionResonanceReceiverReady,
		SourceWeightedAdmissionResonanceIntentReady:                          sourceStage.SourceWeightedAdmissionResonanceIntentReady,
		SourceWeightedAdmissionFinalGateReady:                                sourceStage.SourceWeightedAdmissionFinalGateReady,
		SourceWeightedAdmissionSealReady:                                     sourceStage.SourceWeightedAdmissionSealReady,
		SourceWeightedAdmissionPermitReady:                                   sourceStage.SourceWeightedAdmissionPermitReady,
		SourceWeightedAdmissionAuthorityConsumed:                             sourceStage.SourceWeightedAdmissionAuthorityConsumed,
		SourceWeightedAdmissionAuthorityRequired:                             sourceStage.SourceWeightedAdmissionAuthorityRequired,
		BodySmokeWeighted:                                                    sourceStage.BodySmokeWeighted,
		NanoDirectRunner:                                                     sourceStage.NanoDirectRunner,
		NanoDirectFinalGate:                                                  sourceStage.NanoDirectFinalGate,
		ResonanceGraftAdmissionProof:                                         sourceStage.ResonanceGraftAdmissionProof,
		BoundaryReportFullChain:                                              sourceStage.BoundaryReportFullChain,
		SourceAuthorityGranted:                                               sourceStage.SourceAuthorityGranted,
		AuthorityGranted:                                                     false,
		ContractsReady:                                                       false,
		WriteAllowed:                                                         false,
		AdmissionAllowed:                                                     false,
		LiveAdmissionEnabled:                                                 false,
		MutatesState:                                                         false,
		BodyTarget:                                                           "none",
		Passed:                                                               true,
		Reason:                                                               "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent",
	}
	stage.CausalID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID(stage)
	stage.WriterPreflightHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash(stage)
	stage.ReadBackHash = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBackHash(stage)
	stage.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID = admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightID(stage)
	if stage.CausalID == "" ||
		stage.WriterPreflightHash == "" ||
		stage.ReadBackHash == "" ||
		stage.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID == "" ||
		stage.WriterPreflightHash == stage.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight read-back proof failed")
	}
	raw, err := json.MarshalIndent(stage, "", "  ")
	if err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight marshal failed: %w", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(outputPath, raw, 0600); err != nil {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight write failed: %w", err)
	}
	fmt.Printf("[admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight] pass: resonance_graft_admission_writer_preflight_report=%s resonance_graft_admission_live_stage_report=%s\n", outputPath, liveStagePath)
	return nil
}

func runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightAssert(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight-assert REPORT")
	}
	report, root, err := readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportForAssert(args[0])
	if err != nil {
		return err
	}
	return admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportError(report, root)
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportError(report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport, root map[string]json.RawMessage) error {
	if _, ok := root["schema"]; !ok {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight schema missing")
	}
	if report.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight schema mismatch: got %q want %q", report.Schema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema)
	}
	if report.Status != "shadow_graft_admission_writer_preflight_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight status mismatch: got %q want %q", report.Status, "shadow_graft_admission_writer_preflight_blocked_dry_run")
	}
	if report.Target != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight target mismatch: got %q want %q", report.Target, "live_route_admission_next_step")
	}
	if report.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight target_kind mismatch: got %q want %q", report.TargetKind, "weighted_internal_world_shadow_graft_admission_writer_preflight")
	}
	if report.TargetMode != "closed_writer_preflight_guard_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight target_mode mismatch: got %q want %q", report.TargetMode, "closed_writer_preflight_guard_dry_run")
	}
	if report.Action != "block_weighted_resonance_shadow_graft_admission_live_stage_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight action mismatch: got %q want %q", report.Action, "block_weighted_resonance_shadow_graft_admission_live_stage_blocked_dry_run")
	}
	if report.WriterState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight writer_state mismatch: got %q want %q", report.WriterState, "blocked")
	}
	if report.WriterAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight writer_action mismatch: got %q want %q", report.WriterAction, "reject_blocked_live_stage")
	}
	if report.RollbackState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight rollback_state mismatch: got %q want %q", report.RollbackState, "blocked")
	}
	if report.RollbackAction != "reject_blocked_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight rollback_action mismatch: got %q want %q", report.RollbackAction, "reject_blocked_live_stage")
	}
	if report.StageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight stage_state mismatch: got %q want %q", report.StageState, "blocked")
	}
	if report.StageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight stage_action mismatch: got %q want %q", report.StageAction, "reject_disabled_enable_gate")
	}
	if report.EnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight enable_state mismatch: got %q want %q", report.EnableState, "disabled")
	}
	if report.EnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight enable_action mismatch: got %q want %q", report.EnableAction, "require_operator_key")
	}
	if report.SwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight switch_state mismatch: got %q want %q", report.SwitchState, "disabled")
	}
	if report.SwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight switch_action mismatch: got %q want %q", report.SwitchAction, "hold_pending_live_admission")
	}
	if report.Promotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight promotion mismatch: got %q want %q", report.Promotion, "pending_live_admission")
	}
	if report.ReceiptShape != "weighted_resonance_shadow_graft_admission_writer_preflight_receipt" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight receipt_shape mismatch: got %q want %q", report.ReceiptShape, "weighted_resonance_shadow_graft_admission_writer_preflight_receipt")
	}
	if report.WriterPreflightKind != "shadow_graft_admission_writer_preflight" ||
		report.WriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		report.WriterPreflightStage != "pre_writer_inventory_graft_admission_writer_preflight" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight shape mismatch")
	}
	for _, required := range []struct {
		name  string
		value bool
	}{
		{"weighted_admission_resonance_graft_admission_writer_preflight_ready", report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady},
		{"weighted_admission_resonance_graft_admission_live_stage_consumed", report.WeightedAdmissionResonanceGraftAdmissionLiveStageConsumed},
		{"weighted_admission_resonance_graft_admission_live_stage_required", report.WeightedAdmissionResonanceGraftAdmissionLiveStageRequired},
		{"next_step_blocked_without_resonance_graft_admission_writer_preflight", report.NextStepBlockedWithoutResonanceGraftAdmissionWriterPreflight},
		{"live_stage_verified", report.LiveStageVerified},
		{"live_stage_hash_verified", report.LiveStageHashVerified},
		{"live_stage_read_back_verified", report.LiveStageReadBackVerified},
		{"enable_gate_verified", report.EnableGateVerified},
		{"enable_gate_hash_verified", report.EnableGateHashVerified},
		{"enable_gate_read_back_verified", report.EnableGateReadBackVerified},
		{"switch_verified", report.SwitchVerified},
		{"switch_hash_verified", report.SwitchHashVerified},
		{"switch_read_back_verified", report.SwitchReadBackVerified},
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
		{"requires_writer", report.RequiresWriter},
		{"rollback_required", report.RollbackRequired},
		{"requires_rollback", report.RequiresRollback},
		{"read_only", report.ReadOnly},
		{"replay_only", report.ReplayOnly},
		{"source_weighted_admission_resonance_graft_admission_live_stage_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady},
		{"source_live_stage_admission_required", report.SourceLiveStageAdmissionRequired},
		{"source_live_stage_shadow_only", report.SourceLiveStageShadowOnly},
		{"source_live_stage_dry_run_only", report.SourceLiveStageDryRunOnly},
		{"source_live_stage_live_ready", report.SourceLiveStageLiveReady},
		{"source_live_stage_requires_writer", report.SourceLiveStageRequiresWriter},
		{"source_live_stage_rollback_required", report.SourceLiveStageRollbackRequired},
		{"source_live_stage_requires_rollback", report.SourceLiveStageRequiresRollback},
		{"source_live_stage_read_only", report.SourceLiveStageReadOnly},
		{"source_live_stage_replay_only", report.SourceLiveStageReplayOnly},
		{"source_live_stage_passed", report.SourceLiveStagePassed},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady},
		{"source_enable_gate_admission_required", report.SourceEnableGateAdmissionRequired},
		{"source_enable_gate_shadow_only", report.SourceEnableGateShadowOnly},
		{"source_enable_gate_dry_run_only", report.SourceEnableGateDryRunOnly},
		{"source_enable_gate_live_ready", report.SourceEnableGateLiveReady},
		{"source_enable_gate_rollback_required", report.SourceEnableGateRollbackRequired},
		{"source_enable_gate_read_only", report.SourceEnableGateReadOnly},
		{"source_enable_gate_replay_only", report.SourceEnableGateReplayOnly},
		{"source_enable_gate_passed", report.SourceEnableGatePassed},
		{"source_weighted_admission_resonance_graft_admission_switch_ready", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady},
		{"source_switch_admission_required", report.SourceSwitchAdmissionRequired},
		{"source_switch_shadow_only", report.SourceSwitchShadowOnly},
		{"source_switch_dry_run_only", report.SourceSwitchDryRunOnly},
		{"source_switch_live_ready", report.SourceSwitchLiveReady},
		{"source_switch_rollback_required", report.SourceSwitchRollbackRequired},
		{"source_switch_read_only", report.SourceSwitchReadOnly},
		{"source_switch_replay_only", report.SourceSwitchReplayOnly},
		{"source_switch_passed", report.SourceSwitchPassed},
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
			return fmt.Errorf("weighted admission resonance graft admission writer preflight %s not ready", required.name)
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
		{"writer_ready", report.WriterReady},
		{"rollback_ready", report.RollbackReady},
		{"source_live_stage_graft_allowed", report.SourceLiveStageGraftAllowed},
		{"source_live_stage_raw_dream_text_allowed", report.SourceLiveStageRawDreamTextAllowed},
		{"source_live_stage_raw_dream_text_observed", report.SourceLiveStageRawDreamTextObserved},
		{"source_live_stage_raw_dream_text_forwarded", report.SourceLiveStageRawDreamTextForwarded},
		{"source_live_stage_janus_surface_allowed", report.SourceLiveStageJanusSurfaceAllowed},
		{"source_live_stage_cooc_learning_allowed", report.SourceLiveStageCoocLearningAllowed},
		{"source_live_stage_delta_harvest_allowed", report.SourceLiveStageDeltaHarvestAllowed},
		{"source_live_stage_body_mutation_allowed", report.SourceLiveStageBodyMutationAllowed},
		{"source_live_stage_writer_ready", report.SourceLiveStageWriterReady},
		{"source_live_stage_rollback_ready", report.SourceLiveStageRollbackReady},
		{"source_live_stage_write_allowed", report.SourceLiveStageWriteAllowed},
		{"source_live_stage_admission_allowed", report.SourceLiveStageAdmissionAllowed},
		{"source_live_stage_live_admission_enabled", report.SourceLiveStageLiveAdmissionEnabled},
		{"source_live_stage_mutates_state", report.SourceLiveStageMutatesState},
		{"source_enable_gate_graft_allowed", report.SourceEnableGateGraftAllowed},
		{"source_enable_gate_raw_dream_text_allowed", report.SourceEnableGateRawDreamTextAllowed},
		{"source_enable_gate_raw_dream_text_observed", report.SourceEnableGateRawDreamTextObserved},
		{"source_enable_gate_raw_dream_text_forwarded", report.SourceEnableGateRawDreamTextForwarded},
		{"source_enable_gate_janus_surface_allowed", report.SourceEnableGateJanusSurfaceAllowed},
		{"source_enable_gate_cooc_learning_allowed", report.SourceEnableGateCoocLearningAllowed},
		{"source_enable_gate_delta_harvest_allowed", report.SourceEnableGateDeltaHarvestAllowed},
		{"source_enable_gate_body_mutation_allowed", report.SourceEnableGateBodyMutationAllowed},
		{"source_enable_gate_write_allowed", report.SourceEnableGateWriteAllowed},
		{"source_enable_gate_admission_allowed", report.SourceEnableGateAdmissionAllowed},
		{"source_enable_gate_live_admission_enabled", report.SourceEnableGateLiveAdmissionEnabled},
		{"source_enable_gate_mutates_state", report.SourceEnableGateMutatesState},
		{"source_switch_graft_allowed", report.SourceSwitchGraftAllowed},
		{"source_switch_raw_dream_text_allowed", report.SourceSwitchRawDreamTextAllowed},
		{"source_switch_raw_dream_text_observed", report.SourceSwitchRawDreamTextObserved},
		{"source_switch_raw_dream_text_forwarded", report.SourceSwitchRawDreamTextForwarded},
		{"source_switch_janus_surface_allowed", report.SourceSwitchJanusSurfaceAllowed},
		{"source_switch_cooc_learning_allowed", report.SourceSwitchCoocLearningAllowed},
		{"source_switch_delta_harvest_allowed", report.SourceSwitchDeltaHarvestAllowed},
		{"source_switch_body_mutation_allowed", report.SourceSwitchBodyMutationAllowed},
		{"source_switch_write_allowed", report.SourceSwitchWriteAllowed},
		{"source_switch_admission_allowed", report.SourceSwitchAdmissionAllowed},
		{"source_switch_live_admission_enabled", report.SourceSwitchLiveAdmissionEnabled},
		{"source_switch_mutates_state", report.SourceSwitchMutatesState},
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
			return fmt.Errorf("weighted admission resonance graft admission writer preflight opened %s", closed.name)
		}
	}
	for _, requiredString := range []struct {
		name  string
		value string
	}{
		{"weighted_admission_resonance_graft_admission_writer_preflight_id", report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID},
		{"causal_id", report.CausalID},
		{"writer_preflight_hash", report.WriterPreflightHash},
		{"read_back_hash", report.ReadBackHash},
		{"source_report", report.SourceReport},
		{"source_weighted_admission_resonance_graft_admission_live_stage_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID},
		{"source_weighted_admission_resonance_graft_admission_live_stage_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID},
		{"source_weighted_admission_resonance_graft_admission_live_stage_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash},
		{"source_weighted_admission_resonance_graft_admission_live_stage_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash},
		{"source_weighted_admission_resonance_graft_admission_enable_gate_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack},
		{"source_weighted_admission_resonance_graft_admission_switch_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID},
		{"source_weighted_admission_resonance_graft_admission_switch_causal_id", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID},
		{"source_weighted_admission_resonance_graft_admission_switch_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash},
		{"source_weighted_admission_resonance_graft_admission_switch_read_back_hash", report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack},
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
			return fmt.Errorf("weighted admission resonance graft admission writer preflight %s missing", requiredString.name)
		}
	}
	if report.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_schema mismatch: got %q want %q", report.SourceSchema, admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema)
	}
	if report.SourceStatus != "shadow_graft_admission_live_stage_blocked_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_status mismatch: got %q want %q", report.SourceStatus, "shadow_graft_admission_live_stage_blocked_dry_run")
	}
	if report.SourceTarget != "live_route_admission_next_step" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_target mismatch: got %q want %q", report.SourceTarget, "live_route_admission_next_step")
	}
	if report.SourceStageState != "blocked" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_stage_state mismatch: got %q want %q", report.SourceStageState, "blocked")
	}
	if report.SourceStageAction != "reject_disabled_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_stage_action mismatch: got %q want %q", report.SourceStageAction, "reject_disabled_enable_gate")
	}
	if report.SourceLiveStageReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" ||
		report.SourceLiveStageKind != "shadow_graft_admission_live_stage" ||
		report.SourceLiveStageMode != "closed_enable_gate_live_stage_guard" ||
		report.SourceLiveStageStage != "pre_writer_graft_admission_live_stage" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source live stage shape mismatch")
	}
	if report.SourceLiveStageBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_live_stage_body_target mismatch: got %q want %q", report.SourceLiveStageBodyTarget, "none")
	}
	if report.SourceLiveStageReason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_live_stage_reason mismatch: got %q", report.SourceLiveStageReason)
	}
	if report.StageState != report.SourceStageState || report.StageAction != report.SourceStageAction {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source stage state/action not carried")
	}
	if report.SourceEnableState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_enable_state mismatch: got %q want %q", report.SourceEnableState, "disabled")
	}
	if report.SourceEnableAction != "require_operator_key" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_enable_action mismatch: got %q want %q", report.SourceEnableAction, "require_operator_key")
	}
	if report.SourceEnableGateReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		report.SourceEnableGateKind != "shadow_graft_admission_enable_gate" ||
		report.SourceEnableGateMode != "closed_switch_enable_guard" ||
		report.SourceEnableGateStage != "pre_live_graft_admission_enable_gate" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source enable gate shape mismatch")
	}
	if report.SourceEnableGateBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_enable_gate_body_target mismatch: got %q want %q", report.SourceEnableGateBodyTarget, "none")
	}
	if report.SourceEnableGateReason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_enable_gate_reason mismatch: got %q", report.SourceEnableGateReason)
	}
	if report.EnableState != report.SourceEnableState || report.EnableAction != report.SourceEnableAction {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source enable state/action not carried")
	}
	if report.SourceSwitchState != "disabled" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_switch_state mismatch: got %q want %q", report.SourceSwitchState, "disabled")
	}
	if report.SourceSwitchAction != "hold_pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_switch_action mismatch: got %q want %q", report.SourceSwitchAction, "hold_pending_live_admission")
	}
	if report.SwitchState != report.SourceSwitchState || report.SwitchAction != report.SourceSwitchAction {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source switch state/action not carried")
	}
	if report.SourceSwitchReceiptShape != "weighted_resonance_shadow_graft_admission_switch_receipt" ||
		report.SourceSwitchKind != "shadow_graft_admission_switch" ||
		report.SourceSwitchMode != "closed_promotion_switch_guard" ||
		report.SourceSwitchStage != "pre_live_graft_admission_switch" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source switch shape mismatch")
	}
	if report.SourceSwitchBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_switch_body_target mismatch: got %q want %q", report.SourceSwitchBodyTarget, "none")
	}
	if report.SourceSwitchReason != "weighted resonance shadow graft admission promotion held at disabled switch without mutation" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_switch_reason mismatch: got %q", report.SourceSwitchReason)
	}
	if report.SourcePromotion != "pending_live_admission" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_promotion mismatch: got %q want %q", report.SourcePromotion, "pending_live_admission")
	}
	if report.SourcePromotionAction != "promote_weighted_resonance_shadow_graft_admission_dry_run" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_promotion_action mismatch: got %q want %q", report.SourcePromotionAction, "promote_weighted_resonance_shadow_graft_admission_dry_run")
	}
	if report.SourcePromotionReceiptShape != "weighted_resonance_shadow_graft_admission_promotion_receipt" ||
		report.SourcePromotionKind != "shadow_graft_admission_promotion" ||
		report.SourcePromotionMode != "closed_decision_promotion" ||
		report.SourcePromotionStage != "pre_live_graft_admission_promotion" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source promotion shape mismatch")
	}
	if report.SourcePromotionBodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source_promotion_body_target mismatch: got %q want %q", report.SourcePromotionBodyTarget, "none")
	}
	if report.BodyTarget != "none" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight body_target mismatch: got %q want %q", report.BodyTarget, "none")
	}
	if !strings.HasPrefix(report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID, "weighted-resonance-graft-admission-writer-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.CausalID, "weighted-resonance-graft-admission-writer-preflight-causal-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight causal prefix mismatch")
	}
	if !strings.HasPrefix(report.WriterPreflightHash, "weighted-resonance-graft-admission-writer-preflight-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight hash prefix mismatch")
	}
	if !strings.HasPrefix(report.ReadBackHash, "weighted-resonance-graft-admission-writer-preflight-read-") ||
		report.WriterPreflightHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight read-back mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID, "weighted-resonance-graft-admission-live-stage-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID, "weighted-resonance-graft-admission-live-stage-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash, "weighted-resonance-graft-admission-live-stage-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack, "weighted-resonance-graft-admission-live-stage-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source live stage mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID, "weighted-resonance-graft-admission-enable-gate-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID, "weighted-resonance-graft-admission-enable-gate-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash, "weighted-resonance-graft-admission-enable-gate-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack, "weighted-resonance-graft-admission-enable-gate-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source enable gate mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID, "weighted-resonance-graft-admission-switch-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID, "weighted-resonance-graft-admission-switch-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash, "weighted-resonance-graft-admission-switch-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack, "weighted-resonance-graft-admission-switch-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source switch mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID, "weighted-resonance-graft-admission-promotion-id-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionCausalID, "weighted-resonance-graft-admission-promotion-causal-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionHash, "weighted-resonance-graft-admission-promotion-") ||
		!strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionPromotionReadBack, "weighted-resonance-graft-admission-promotion-read-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source promotion mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID, "weighted-resonance-graft-admission-decision-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source decision id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofPreconditionID, "weighted-resonance-graft-admission-proof-precondition-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source precondition id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftAdmissionProofID, "weighted-resonance-graft-admission-proof-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source proof id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID, "weighted-resonance-graft-candidate-store-reader-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source reader id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateStoreID, "weighted-resonance-graft-candidate-store-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source store id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftCandidateID, "weighted-resonance-graft-candidate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source candidate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftGateID, "weighted-resonance-graft-gate-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source gate id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftPreflightID, "weighted-resonance-graft-preflight-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source preflight id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceGraftBoundaryID, "weighted-resonance-graft-boundary-id-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source boundary id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceObservationID, "weighted-resonance-observation-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source observation id prefix mismatch")
	}
	if !strings.HasPrefix(report.SourceWeightedAdmissionResonanceReceiverID, "weighted-resonance-receiver-") {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight source receiver id prefix mismatch")
	}
	if report.CausalID == "" || report.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight causal_id mismatch")
	}
	if report.WriterPreflightHash == "" || report.WriterPreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight writer_preflight_hash mismatch")
	}
	if report.ReadBackHash == "" || report.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBackHash(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight read_back_hash mismatch")
	}
	if report.WriterPreflightHash == report.ReadBackHash {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight read-back proof collapsed")
	}
	if report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID == "" || report.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightID(report) {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight id mismatch")
	}
	if report.Reason != "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		return fmt.Errorf("weighted admission resonance graft admission writer preflight reason mismatch: got %q", report.Reason)
	}
	return nil
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport) string {
	h := hashJSON(struct {
		SourceLiveStageID    string `json:"source_live_stage_id"`
		SourceLiveStageRead  string `json:"source_live_stage_read_back_hash"`
		SourceEnableGateID   string `json:"source_enable_gate_id"`
		Target               string `json:"target"`
		WriterPreflightKind  string `json:"writer_preflight_kind"`
		WriterPreflightStage string `json:"writer_preflight_stage"`
	}{
		SourceLiveStageID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceLiveStageRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceEnableGateID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		Target:               sw.Target,
		WriterPreflightKind:  sw.WriterPreflightKind,
		WriterPreflightStage: sw.WriterPreflightStage,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-writer-preflight-causal-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport) string {
	h := hashJSON(struct {
		CausalID            string `json:"causal_id"`
		SourceLiveStageID   string `json:"source_live_stage_id"`
		SourceLiveStageHash string `json:"source_live_stage_hash"`
		SourceLiveStageRead string `json:"source_live_stage_read_back_hash"`
		WriterState         string `json:"writer_state"`
		WriterAction        string `json:"writer_action"`
		RollbackState       string `json:"rollback_state"`
		RollbackAction      string `json:"rollback_action"`
		StageState          string `json:"stage_state"`
		StageAction         string `json:"stage_action"`
		EnableState         string `json:"enable_state"`
		EnableAction        string `json:"enable_action"`
		SwitchState         string `json:"switch_state"`
		SwitchAction        string `json:"switch_action"`
		Promotion           string `json:"promotion"`
		Action              string `json:"action"`
		ReceiptShape        string `json:"receipt_shape"`
		WriterPreflightMode string `json:"writer_preflight_mode"`
		LiveStageVerified   bool   `json:"live_stage_verified"`
		RequiresWriter      bool   `json:"requires_writer"`
		WriterReady         bool   `json:"writer_ready"`
		RequiresRollback    bool   `json:"requires_rollback"`
		RollbackReady       bool   `json:"rollback_ready"`
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
		SourceLiveStageID:   sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceLiveStageHash: sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceLiveStageRead: sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		WriterState:         sw.WriterState,
		WriterAction:        sw.WriterAction,
		RollbackState:       sw.RollbackState,
		RollbackAction:      sw.RollbackAction,
		StageState:          sw.StageState,
		StageAction:         sw.StageAction,
		EnableState:         sw.EnableState,
		EnableAction:        sw.EnableAction,
		SwitchState:         sw.SwitchState,
		SwitchAction:        sw.SwitchAction,
		Promotion:           sw.Promotion,
		Action:              sw.Action,
		ReceiptShape:        sw.ReceiptShape,
		WriterPreflightMode: sw.WriterPreflightMode,
		LiveStageVerified:   sw.LiveStageVerified,
		RequiresWriter:      sw.RequiresWriter,
		WriterReady:         sw.WriterReady,
		RequiresRollback:    sw.RequiresRollback,
		RollbackReady:       sw.RollbackReady,
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
	return "weighted-resonance-graft-admission-writer-preflight-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBackHash(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport) string {
	h := hashJSON(struct {
		WriterPreflightHash  string `json:"writer_preflight_hash"`
		SourceLiveStageID    string `json:"source_live_stage_id"`
		SourceLiveStageRead  string `json:"source_live_stage_read_back_hash"`
		WriterPreflightKind  string `json:"writer_preflight_kind"`
		WriterPreflightReady bool   `json:"writer_preflight_ready"`
		LiveStageConsumed    bool   `json:"live_stage_consumed"`
		RequiresWriter       bool   `json:"requires_writer"`
		WriterReady          bool   `json:"writer_ready"`
		RequiresRollback     bool   `json:"requires_rollback"`
		RollbackReady        bool   `json:"rollback_ready"`
		LiveReady            bool   `json:"live_ready"`
		BodyMutation         bool   `json:"body_mutation"`
		LiveAdmission        bool   `json:"live_admission"`
		WriteAllowed         bool   `json:"write_allowed"`
		AdmissionAllowed     bool   `json:"admission_allowed"`
	}{
		WriterPreflightHash:  sw.WriterPreflightHash,
		SourceLiveStageID:    sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceLiveStageRead:  sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		WriterPreflightKind:  sw.WriterPreflightKind,
		WriterPreflightReady: sw.WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady,
		LiveStageConsumed:    sw.WeightedAdmissionResonanceGraftAdmissionLiveStageConsumed,
		RequiresWriter:       sw.RequiresWriter,
		WriterReady:          sw.WriterReady,
		RequiresRollback:     sw.RequiresRollback,
		RollbackReady:        sw.RollbackReady,
		LiveReady:            sw.LiveReady,
		BodyMutation:         sw.BodyMutationAllowed,
		LiveAdmission:        sw.LiveAdmissionEnabled,
		WriteAllowed:         sw.WriteAllowed,
		AdmissionAllowed:     sw.AdmissionAllowed,
	})
	if h == "" {
		return ""
	}
	return "weighted-resonance-graft-admission-writer-preflight-read-" + h
}

func admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightID(sw admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport) string {
	h := hashJSON(struct {
		Schema                  string `json:"schema"`
		Status                  string `json:"status"`
		Action                  string `json:"action"`
		WriterState             string `json:"writer_state"`
		WriterAction            string `json:"writer_action"`
		RollbackState           string `json:"rollback_state"`
		RollbackAction          string `json:"rollback_action"`
		StageState              string `json:"stage_state"`
		StageAction             string `json:"stage_action"`
		EnableState             string `json:"enable_state"`
		EnableAction            string `json:"enable_action"`
		SwitchState             string `json:"switch_state"`
		SwitchAction            string `json:"switch_action"`
		Promotion               string `json:"promotion"`
		SourceReport            string `json:"source_report"`
		SourceLiveStageID       string `json:"source_live_stage_id"`
		SourceLiveStageHash     string `json:"source_live_stage_hash"`
		SourceLiveStageRead     string `json:"source_live_stage_read_back_hash"`
		SourceEnableGateID      string `json:"source_enable_gate_id"`
		SourceEnableGateHash    string `json:"source_enable_gate_hash"`
		SourceEnableGateRead    string `json:"source_enable_gate_read_back_hash"`
		SourceSwitchID          string `json:"source_switch_id"`
		SourceSwitchHash        string `json:"source_switch_hash"`
		SourceSwitchRead        string `json:"source_switch_read_back_hash"`
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
		WriterPreflightHash     string `json:"writer_preflight_hash"`
		ReadBackHash            string `json:"read_back_hash"`
		Ready                   bool   `json:"ready"`
		ReceiptShape            string `json:"receipt_shape"`
		WriterPreflightKind     string `json:"writer_preflight_kind"`
		WriterPreflightMode     string `json:"writer_preflight_mode"`
		WriterPreflightStage    string `json:"writer_preflight_stage"`
		LiveStageVerified       bool   `json:"live_stage_verified"`
		EnableGateVerified      bool   `json:"enable_gate_verified"`
		SwitchVerified          bool   `json:"switch_verified"`
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
		RequiresWriter          bool   `json:"requires_writer"`
		WriterReady             bool   `json:"writer_ready"`
		RollbackRequired        bool   `json:"rollback_required"`
		RequiresRollback        bool   `json:"requires_rollback"`
		RollbackReady           bool   `json:"rollback_ready"`
		ReadOnly                bool   `json:"read_only"`
		ReplayOnly              bool   `json:"replay_only"`
		LiveReady               bool   `json:"live_ready"`
		ContractsReady          bool   `json:"contracts_ready"`
		BodyTarget              string `json:"body_target"`
		WriteAllowed            bool   `json:"write_allowed"`
		AdmissionAllowed        bool   `json:"admission_allowed"`
		LiveAdmissionEnabled    bool   `json:"live_admission_enabled"`
		MutatesState            bool   `json:"mutates_state"`
		NextStepBlockedWithout  bool   `json:"next_step_blocked_without_resonance_graft_admission_writer_preflight"`
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
		WriterState:             sw.WriterState,
		WriterAction:            sw.WriterAction,
		RollbackState:           sw.RollbackState,
		RollbackAction:          sw.RollbackAction,
		StageState:              sw.StageState,
		StageAction:             sw.StageAction,
		EnableState:             sw.EnableState,
		EnableAction:            sw.EnableAction,
		SwitchState:             sw.SwitchState,
		SwitchAction:            sw.SwitchAction,
		Promotion:               sw.Promotion,
		SourceReport:            sw.SourceReport,
		SourceLiveStageID:       sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID,
		SourceLiveStageHash:     sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash,
		SourceLiveStageRead:     sw.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack,
		SourceEnableGateID:      sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID,
		SourceEnableGateHash:    sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash,
		SourceEnableGateRead:    sw.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack,
		SourceSwitchID:          sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID,
		SourceSwitchHash:        sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash,
		SourceSwitchRead:        sw.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack,
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
		WriterPreflightHash:     sw.WriterPreflightHash,
		ReadBackHash:            sw.ReadBackHash,
		Ready:                   sw.WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady,
		ReceiptShape:            sw.ReceiptShape,
		WriterPreflightKind:     sw.WriterPreflightKind,
		WriterPreflightMode:     sw.WriterPreflightMode,
		WriterPreflightStage:    sw.WriterPreflightStage,
		LiveStageVerified:       sw.LiveStageVerified,
		EnableGateVerified:      sw.EnableGateVerified,
		SwitchVerified:          sw.SwitchVerified,
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
		RequiresWriter:          sw.RequiresWriter,
		WriterReady:             sw.WriterReady,
		RollbackRequired:        sw.RollbackRequired,
		RequiresRollback:        sw.RequiresRollback,
		RollbackReady:           sw.RollbackReady,
		ReadOnly:                sw.ReadOnly,
		ReplayOnly:              sw.ReplayOnly,
		LiveReady:               sw.LiveReady,
		ContractsReady:          sw.ContractsReady,
		BodyTarget:              sw.BodyTarget,
		WriteAllowed:            sw.WriteAllowed,
		AdmissionAllowed:        sw.AdmissionAllowed,
		LiveAdmissionEnabled:    sw.LiveAdmissionEnabled,
		MutatesState:            sw.MutatesState,
		NextStepBlockedWithout:  sw.NextStepBlockedWithoutResonanceGraftAdmissionWriterPreflight,
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
	return "weighted-resonance-graft-admission-writer-preflight-id-" + h
}

func readAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReportForAssert(path string) (admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport, map[string]json.RawMessage, error) {
	var report admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport
	if strings.TrimSpace(path) == "" {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer preflight path missing")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return report, nil, fmt.Errorf("weighted admission resonance graft admission writer preflight not written")
		}
		return report, nil, err
	}
	if len(raw) == 0 {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer preflight not written")
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(raw, &root); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer preflight JSON invalid: %w", err)
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, nil, fmt.Errorf("weighted admission resonance graft admission writer preflight decode failed: %w", err)
	}
	return report, root, nil
}
