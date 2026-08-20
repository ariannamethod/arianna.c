package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{"live_stage.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{"live_stage.json", "writer_preflight.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-writer-preflight RESONANCE_GRAFT_ADMISSION_LIVE_STAGE_REPORT RESONANCE_GRAFT_ADMISSION_WRITER_PREFLIGHT_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{"  ", filepath.Join(dir, "writer_preflight.json")}),
		"weighted admission resonance graft admission live stage path missing",
	)

	liveStagePath := filepath.Join(dir, "live_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, liveStagePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{liveStagePath, "  "}),
		"weighted admission resonance graft admission writer preflight output path missing",
	)

	writerPreflightPath := filepath.Join(dir, "writer_preflight.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{liveStagePath, writerPreflightPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission writer preflight rejected: %v", err)
	}
	raw, err := os.ReadFile(writerPreflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission writer preflight: %v", err)
	}
	var preflight admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReport
	if err := json.Unmarshal(raw, &preflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission writer preflight: %v", err)
	}
	liveStageRaw, err := os.ReadFile(liveStagePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission live stage: %v", err)
	}
	var sourceStage admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageReport
	if err := json.Unmarshal(liveStageRaw, &sourceStage); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission live stage: %v", err)
	}
	if preflight.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightSchema ||
		preflight.Status != "shadow_graft_admission_writer_preflight_blocked_dry_run" ||
		preflight.Target != "live_route_admission_next_step" ||
		preflight.TargetKind != "weighted_internal_world_shadow_graft_admission_writer_preflight" ||
		preflight.TargetMode != "closed_writer_preflight_guard_dry_run" ||
		preflight.Action != "block_weighted_resonance_shadow_graft_admission_live_stage_blocked_dry_run" ||
		preflight.WriterState != "blocked" ||
		preflight.WriterAction != "reject_blocked_live_stage" ||
		preflight.RollbackState != "blocked" ||
		preflight.RollbackAction != "reject_blocked_live_stage" ||
		preflight.StageState != "blocked" ||
		preflight.StageAction != "reject_disabled_enable_gate" ||
		preflight.EnableState != "disabled" ||
		preflight.EnableAction != "require_operator_key" ||
		preflight.SwitchState != "disabled" ||
		preflight.SwitchAction != "hold_pending_live_admission" ||
		preflight.Promotion != "pending_live_admission" ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionWriterPreflightReady ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionLiveStageConsumed ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionLiveStageRequired ||
		!preflight.NextStepBlockedWithoutResonanceGraftAdmissionWriterPreflight ||
		preflight.ReceiptShape != "weighted_resonance_shadow_graft_admission_writer_preflight_receipt" ||
		preflight.WriterPreflightKind != "shadow_graft_admission_writer_preflight" ||
		preflight.WriterPreflightMode != "closed_live_stage_writer_preflight_guard" ||
		preflight.WriterPreflightStage != "pre_writer_inventory_graft_admission_writer_preflight" ||
		preflight.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightCausalID(preflight) ||
		preflight.WriterPreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightHash(preflight) ||
		preflight.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightReadBackHash(preflight) ||
		preflight.WriterPreflightHash == preflight.ReadBackHash ||
		preflight.WeightedAdmissionResonanceGraftAdmissionWriterPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflightID(preflight) ||
		!preflight.LiveStageVerified ||
		!preflight.LiveStageHashVerified ||
		!preflight.LiveStageReadBackVerified ||
		!preflight.EnableGateVerified ||
		!preflight.EnableGateHashVerified ||
		!preflight.EnableGateReadBackVerified ||
		!preflight.SwitchVerified ||
		!preflight.SwitchHashVerified ||
		!preflight.SwitchReadBackVerified ||
		!preflight.PromotionVerified ||
		!preflight.DecisionVerified ||
		!preflight.ProofVerified ||
		!preflight.StoreReaderVerified ||
		!preflight.CandidateVerified ||
		!preflight.AuthorityVerified ||
		!preflight.AdmissionRequired ||
		!preflight.ShadowOnly ||
		preflight.GraftAllowed ||
		!preflight.DryRunOnly ||
		!preflight.LiveReady ||
		preflight.RawDreamTextAllowed ||
		preflight.JanusSurfaceAllowed ||
		preflight.CoocLearningAllowed ||
		preflight.DeltaHarvestAllowed ||
		preflight.BodyMutationAllowed ||
		!preflight.RequiresWriter ||
		preflight.WriterReady ||
		!preflight.RollbackRequired ||
		!preflight.RequiresRollback ||
		preflight.RollbackReady ||
		!preflight.ReadOnly ||
		!preflight.ReplayOnly ||
		preflight.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema ||
		preflight.SourceStatus != "shadow_graft_admission_live_stage_blocked_dry_run" ||
		preflight.SourceTarget != "live_route_admission_next_step" ||
		preflight.SourceReport != liveStagePath ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageID != sourceStage.WeightedAdmissionResonanceGraftAdmissionLiveStageID ||
		!preflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReady ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageCausalID != sourceStage.CausalID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageHash != sourceStage.LiveStageHash ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionLiveStageReadBack != sourceStage.ReadBackHash ||
		preflight.SourceStageState != sourceStage.StageState ||
		preflight.SourceStageAction != sourceStage.StageAction ||
		preflight.SourceLiveStageKind != sourceStage.LiveStageKind ||
		preflight.SourceLiveStageMode != sourceStage.LiveStageMode ||
		preflight.SourceLiveStageStage != sourceStage.LiveStageStage ||
		preflight.SourceLiveStageGraftAllowed ||
		preflight.SourceLiveStageWriterReady ||
		preflight.SourceLiveStageRollbackReady ||
		preflight.SourceLiveStageWriteAllowed ||
		preflight.SourceLiveStageAdmissionAllowed ||
		preflight.SourceLiveStageLiveAdmissionEnabled ||
		preflight.SourceLiveStageMutatesState ||
		preflight.SourceLiveStageBodyTarget != "none" ||
		!preflight.SourceLiveStagePassed ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID != sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionEnableGateID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID != sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID != sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID != sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionProofID != sourceStage.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		preflight.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != sourceStage.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		!preflight.BodySmokeWeighted ||
		!preflight.NanoDirectRunner ||
		!preflight.NanoDirectFinalGate ||
		!preflight.ResonanceGraftAdmissionProof ||
		!preflight.BoundaryReportFullChain ||
		preflight.SourceAuthorityGranted ||
		preflight.AuthorityGranted ||
		preflight.ContractsReady ||
		preflight.WriteAllowed ||
		preflight.AdmissionAllowed ||
		preflight.LiveAdmissionEnabled ||
		preflight.MutatesState ||
		preflight.BodyTarget != "none" ||
		!preflight.Passed ||
		preflight.Reason != "weighted resonance shadow graft admission writer preflight blocked by blocked live stage; writer and rollback remain absent" {
		t.Fatalf("weighted admission resonance graft admission writer preflight lost contract: %+v", preflight)
	}

	openedStagePath := filepath.Join(dir, "opened_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, openedStagePath)
	writeWeightedReadinessFixture(t, openedStagePath, stringsReplaceFirst(readText(t, openedStagePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{openedStagePath, filepath.Join(dir, "opened_writer_preflight.json")}),
		"weighted admission resonance graft admission live stage opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_live_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{badSchemaPath, filepath.Join(dir, "bad_schema_writer_preflight.json")}),
		`weighted admission resonance graft admission live stage schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_live_stage.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionLiveStageSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_live_stage.json")
	writeWeightedAdmissionResonanceGraftAdmissionLiveStageFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"live_stage_hash": "weighted-resonance-graft-admission-live-stage-`, `"live_stage_hash": "weighted-resonance-graft-admission-live-stage-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{badHashPath, filepath.Join(dir, "bad_hash_writer_preflight.json")}),
		"weighted admission resonance graft admission live stage live_stage_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionWriterPreflight([]string{liveStagePath, filepath.Join(dir, "missing", "writer_preflight.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission writer preflight write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission writer preflight write failure, got %v", err)
	}
}
