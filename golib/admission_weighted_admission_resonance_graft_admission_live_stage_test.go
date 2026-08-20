package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{"enable_gate.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{"enable_gate.json", "live_stage.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-live-stage RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{"  ", filepath.Join(dir, "live_stage.json")}),
		"weighted admission resonance graft admission enable gate path missing",
	)

	enableGatePath := filepath.Join(dir, "enable_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, enableGatePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{enableGatePath, "  "}),
		"weighted admission resonance graft admission live stage output path missing",
	)

	liveStagePath := filepath.Join(dir, "live_stage.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{enableGatePath, liveStagePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission live stage rejected: %v", err)
	}
	raw, err := os.ReadFile(liveStagePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission live stage: %v", err)
	}
	var stage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport
	if err := json.Unmarshal(raw, &stage); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission live stage: %v", err)
	}
	enableGateRaw, err := os.ReadFile(enableGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission enable gate: %v", err)
	}
	var sourceGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport
	if err := json.Unmarshal(enableGateRaw, &sourceGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission enable gate: %v", err)
	}
	if stage.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema ||
		stage.Status != "shadow_graft_admission_live_stage_blocked_dry_run" ||
		stage.Target != "live_route_admission_next_step" ||
		stage.TargetKind != "weighted_internal_world_shadow_graft_admission_live_stage" ||
		stage.TargetMode != "closed_live_stage_guard_dry_run" ||
		stage.Action != "block_weighted_resonance_shadow_graft_admission_enable_gate_disabled_dry_run" ||
		stage.StageState != "blocked" ||
		stage.StageAction != "reject_disabled_enable_gate" ||
		stage.EnableState != "disabled" ||
		stage.EnableAction != "require_operator_key" ||
		stage.SwitchState != "disabled" ||
		stage.SwitchAction != "hold_pending_live_admission" ||
		stage.Promotion != "pending_live_admission" ||
		!stage.WeightedAdmissionResonanceGraftAdmissionLiveStageReady ||
		!stage.WeightedAdmissionResonanceGraftAdmissionEnableGateConsumed ||
		!stage.WeightedAdmissionResonanceGraftAdmissionEnableGateRequired ||
		!stage.NextStepBlockedWithoutResonanceGraftAdmissionLiveStage ||
		stage.ReceiptShape != "weighted_resonance_shadow_graft_admission_live_stage_receipt" ||
		stage.LiveStageKind != "shadow_graft_admission_live_stage" ||
		stage.LiveStageMode != "closed_enable_gate_live_stage_guard" ||
		stage.LiveStageStage != "pre_writer_graft_admission_live_stage" ||
		stage.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID(stage) ||
		stage.LiveStageHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageHash(stage) ||
		stage.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReadBackHash(stage) ||
		stage.LiveStageHash == stage.ReadBackHash ||
		stage.WeightedAdmissionResonanceGraftAdmissionLiveStageID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageID(stage) ||
		!stage.EnableGateVerified ||
		!stage.EnableGateHashVerified ||
		!stage.EnableGateReadBackVerified ||
		!stage.SwitchVerified ||
		!stage.SwitchHashVerified ||
		!stage.SwitchReadBackVerified ||
		!stage.PromotionVerified ||
		!stage.DecisionVerified ||
		!stage.ProofVerified ||
		!stage.StoreReaderVerified ||
		!stage.CandidateVerified ||
		!stage.AuthorityVerified ||
		!stage.AdmissionRequired ||
		!stage.ShadowOnly ||
		stage.GraftAllowed ||
		!stage.DryRunOnly ||
		!stage.LiveReady ||
		stage.RawDreamTextAllowed ||
		stage.JanusSurfaceAllowed ||
		stage.CoocLearningAllowed ||
		stage.DeltaHarvestAllowed ||
		stage.BodyMutationAllowed ||
		!stage.RequiresWriter ||
		stage.WriterReady ||
		!stage.RollbackRequired ||
		!stage.RequiresRollback ||
		stage.RollbackReady ||
		!stage.ReadOnly ||
		!stage.ReplayOnly ||
		stage.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema ||
		stage.SourceStatus != "shadow_graft_admission_enable_gate_disabled_dry_run" ||
		stage.SourceTarget != "live_route_admission_next_step" ||
		stage.SourceReport != enableGatePath ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID != sourceGate.WeightedAdmissionResonanceGraftAdmissionEnableGateID ||
		!stage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReady ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID != sourceGate.CausalID ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateHash != sourceGate.EnableGateHash ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateReadBack != sourceGate.ReadBackHash ||
		stage.SourceEnableState != sourceGate.EnableState ||
		stage.SourceEnableAction != sourceGate.EnableAction ||
		stage.SourceEnableGateKind != sourceGate.EnableGateKind ||
		stage.SourceEnableGateMode != sourceGate.EnableGateMode ||
		stage.SourceEnableGateStage != sourceGate.EnableGateStage ||
		stage.SourceEnableGateGraftAllowed ||
		stage.SourceEnableGateWriteAllowed ||
		stage.SourceEnableGateAdmissionAllowed ||
		stage.SourceEnableGateLiveAdmissionEnabled ||
		stage.SourceEnableGateMutatesState ||
		stage.SourceEnableGateBodyTarget != "none" ||
		!stage.SourceEnableGatePassed ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID != sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID != sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID != sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID ||
		stage.SourceWeightedAdmissionResonanceGraftAdmissionProofID != sourceGate.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		stage.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != sourceGate.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		!stage.BodySmokeWeighted ||
		!stage.NanoDirectRunner ||
		!stage.NanoDirectFinalGate ||
		!stage.ResonanceGraftAdmissionProof ||
		!stage.BoundaryReportFullChain ||
		stage.SourceAuthorityGranted ||
		stage.AuthorityGranted ||
		stage.ContractsReady ||
		stage.WriteAllowed ||
		stage.AdmissionAllowed ||
		stage.LiveAdmissionEnabled ||
		stage.MutatesState ||
		stage.BodyTarget != "none" ||
		!stage.Passed ||
		stage.Reason != "weighted resonance shadow graft admission live stage blocked by disabled enable gate; writer and rollback remain absent" {
		t.Fatalf("weighted admission resonance graft admission live stage lost contract: %+v", stage)
	}

	openedGatePath := filepath.Join(dir, "opened_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, openedGatePath)
	writeWeightedReadinessFixture(t, openedGatePath, stringsReplaceFirst(readText(t, openedGatePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{openedGatePath, filepath.Join(dir, "opened_live_stage.json")}),
		"weighted admission resonance graft admission enable gate opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_enable_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{badSchemaPath, filepath.Join(dir, "bad_schema_live_stage.json")}),
		`weighted admission resonance graft admission enable gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_enable_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_enable_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionEnableGateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-`, `"enable_gate_hash": "weighted-resonance-graft-admission-enable-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{badHashPath, filepath.Join(dir, "bad_hash_live_stage.json")}),
		"weighted admission resonance graft admission enable gate enable_gate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStage([]string{enableGatePath, filepath.Join(dir, "missing", "live_stage.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission live stage write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission live stage write failure, got %v", err)
	}
}
