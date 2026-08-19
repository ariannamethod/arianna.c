package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{"switch.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{"switch.json", "enable_gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-admission-enable-gate RESONANCE_GRAFT_ADMISSION_SWITCH_REPORT RESONANCE_GRAFT_ADMISSION_ENABLE_GATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{"  ", filepath.Join(dir, "enable_gate.json")}),
		"weighted admission resonance graft admission switch path missing",
	)

	switchPath := filepath.Join(dir, "switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, switchPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{switchPath, "  "}),
		"weighted admission resonance graft admission enable gate output path missing",
	)

	enableGatePath := filepath.Join(dir, "enable_gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{switchPath, enableGatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission enable gate rejected: %v", err)
	}
	raw, err := os.ReadFile(enableGatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission enable gate: %v", err)
	}
	var gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReport
	if err := json.Unmarshal(raw, &gate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission enable gate: %v", err)
	}
	switchRaw, err := os.ReadFile(switchPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission switch: %v", err)
	}
	var sourceSwitch admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchReport
	if err := json.Unmarshal(switchRaw, &sourceSwitch); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission switch: %v", err)
	}
	if gate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateSchema ||
		gate.Status != "shadow_graft_admission_enable_gate_disabled_dry_run" ||
		gate.Target != "live_route_admission_next_step" ||
		gate.TargetKind != "weighted_internal_world_shadow_graft_admission_enable_gate" ||
		gate.TargetMode != "closed_enable_gate_dry_run" ||
		gate.Action != "hold_weighted_resonance_shadow_graft_admission_switch_disabled_dry_run" ||
		gate.EnableState != "disabled" ||
		gate.EnableAction != "require_operator_key" ||
		gate.SwitchState != "disabled" ||
		gate.SwitchAction != "hold_pending_live_admission" ||
		gate.Promotion != "pending_live_admission" ||
		!gate.WeightedAdmissionResonanceGraftAdmissionEnableGateReady ||
		!gate.WeightedAdmissionResonanceGraftAdmissionSwitchConsumed ||
		!gate.WeightedAdmissionResonanceGraftAdmissionSwitchRequired ||
		!gate.NextStepBlockedWithoutResonanceGraftAdmissionEnableGate ||
		gate.ReceiptShape != "weighted_resonance_shadow_graft_admission_enable_gate_receipt" ||
		gate.EnableGateKind != "shadow_graft_admission_enable_gate" ||
		gate.EnableGateMode != "closed_switch_enable_guard" ||
		gate.EnableGateStage != "pre_live_graft_admission_enable_gate" ||
		gate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateCausalID(gate) ||
		gate.EnableGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateHash(gate) ||
		gate.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateReadBackHash(gate) ||
		gate.EnableGateHash == gate.ReadBackHash ||
		gate.WeightedAdmissionResonanceGraftAdmissionEnableGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGateID(gate) ||
		!gate.SwitchVerified ||
		!gate.SwitchHashVerified ||
		!gate.SwitchReadBackVerified ||
		!gate.PromotionVerified ||
		!gate.DecisionVerified ||
		!gate.ProofVerified ||
		!gate.StoreReaderVerified ||
		!gate.CandidateVerified ||
		!gate.AuthorityVerified ||
		!gate.AdmissionRequired ||
		!gate.ShadowOnly ||
		gate.GraftAllowed ||
		!gate.DryRunOnly ||
		!gate.LiveReady ||
		gate.RawDreamTextAllowed ||
		gate.JanusSurfaceAllowed ||
		gate.CoocLearningAllowed ||
		gate.DeltaHarvestAllowed ||
		gate.BodyMutationAllowed ||
		!gate.RollbackRequired ||
		!gate.ReadOnly ||
		!gate.ReplayOnly ||
		gate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema ||
		gate.SourceStatus != "shadow_graft_admission_switch_disabled_dry_run" ||
		gate.SourceTarget != "live_route_admission_next_step" ||
		gate.SourceReport != switchPath ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchID != sourceSwitch.WeightedAdmissionResonanceGraftAdmissionSwitchID ||
		!gate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReady ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchCausalID != sourceSwitch.CausalID ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchHash != sourceSwitch.SwitchHash ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionSwitchReadBack != sourceSwitch.ReadBackHash ||
		gate.SourceSwitchState != sourceSwitch.SwitchState ||
		gate.SourceSwitchAction != sourceSwitch.SwitchAction ||
		gate.SourceSwitchKind != sourceSwitch.SwitchKind ||
		gate.SourceSwitchMode != sourceSwitch.SwitchMode ||
		gate.SourceSwitchStage != sourceSwitch.SwitchStage ||
		gate.SourceSwitchGraftAllowed ||
		gate.SourceSwitchWriteAllowed ||
		gate.SourceSwitchAdmissionAllowed ||
		gate.SourceSwitchLiveAdmissionEnabled ||
		gate.SourceSwitchMutatesState ||
		gate.SourceSwitchBodyTarget != "none" ||
		!gate.SourceSwitchPassed ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID != sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionPromotionID ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID != sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionDecisionID ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionProofID != sourceSwitch.SourceWeightedAdmissionResonanceGraftAdmissionProofID ||
		gate.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID != sourceSwitch.SourceWeightedAdmissionResonanceGraftCandidateStoreReaderID ||
		!gate.BodySmokeWeighted ||
		!gate.NanoDirectRunner ||
		!gate.NanoDirectFinalGate ||
		!gate.ResonanceGraftAdmissionProof ||
		!gate.BoundaryReportFullChain ||
		gate.SourceAuthorityGranted ||
		gate.AuthorityGranted ||
		gate.ContractsReady ||
		gate.WriteAllowed ||
		gate.AdmissionAllowed ||
		gate.LiveAdmissionEnabled ||
		gate.MutatesState ||
		gate.BodyTarget != "none" ||
		!gate.Passed ||
		gate.Reason != "weighted resonance shadow graft admission enable gate closed; operator key absent and mutation refused" {
		t.Fatalf("weighted admission resonance graft admission enable gate lost contract: %+v", gate)
	}

	openedSwitchPath := filepath.Join(dir, "opened_switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, openedSwitchPath)
	writeWeightedReadinessFixture(t, openedSwitchPath, stringsReplaceFirst(readText(t, openedSwitchPath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{openedSwitchPath, filepath.Join(dir, "opened_enable_gate.json")}),
		"weighted admission resonance graft admission switch opened live_admission_enabled",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_enable_gate.json")}),
		`weighted admission resonance graft admission switch schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_admission_switch.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionSwitchSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_switch.json")
	writeWeightedAdmissionResonanceGraftAdmissionSwitchFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"switch_hash": "weighted-resonance-graft-admission-switch-`, `"switch_hash": "weighted-resonance-graft-admission-switch-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{badHashPath, filepath.Join(dir, "bad_hash_enable_gate.json")}),
		"weighted admission resonance graft admission switch switch_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionEnableGate([]string{switchPath, filepath.Join(dir, "missing", "enable_gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission enable gate write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission enable gate write failure, got %v", err)
	}
}
