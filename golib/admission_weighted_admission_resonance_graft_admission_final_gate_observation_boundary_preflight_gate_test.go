package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{"preflight.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{"preflight.json", "gate.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{"  ", filepath.Join(dir, "gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight path missing",
	)

	preflightPath := filepath.Join(dir, "preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, preflightPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{preflightPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate output path missing",
	)

	gatePath := filepath.Join(dir, "gate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{preflightPath, gatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate rejected: %v", err)
	}
	raw, err := os.ReadFile(gatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate: %v", err)
	}
	var gate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport
	if err := json.Unmarshal(raw, &gate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate: %v", err)
	}
	sourceRaw, err := os.ReadFile(preflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight: %v", err)
	}
	var sourcePreflight admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReport
	if err := json.Unmarshal(sourceRaw, &sourcePreflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight: %v", err)
	}
	if gate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema ||
		gate.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run" ||
		gate.Target != "live_route_admission_next_step" ||
		gate.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate" ||
		gate.TargetMode != "closed_preflight_gate_guard_dry_run" ||
		gate.Action != "gate_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run" ||
		gate.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" ||
		gate.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" ||
		gate.LedgerState != "blocked" ||
		gate.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate" ||
		gate.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_receipt" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateState != "blocked" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateAction != "gate_blocked_final_gate_observation_boundary_preflight" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateTarget != "resonance" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight" ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateTargetMode != "closed_preflight_gate_guard_dry_run" ||
		!gate.AdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly ||
		!gate.AdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified ||
		!gate.AdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified ||
		!gate.AdmissionFinalGateObservationBoundaryPreflightGateObservationVerified ||
		!gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateReady ||
		gate.FinalGateObservationBoundaryPreflightGateKind != "blocked_final_gate_observation_boundary_preflight_gate" ||
		gate.FinalGateObservationBoundaryPreflightGateMode != "no_mutation_preflight_gate" ||
		gate.FinalGateObservationBoundaryPreflightGateStage != "post_boundary_preflight_pre_live_admission" ||
		gate.FinalGateObservationBoundaryPreflightGateRawDreamTextObserved ||
		gate.FinalGateObservationBoundaryPreflightGateRawDreamTextForwarded ||
		gate.FinalGateObservationBoundaryPreflightGateRawDreamTextAllowed ||
		gate.FinalGateObservationBoundaryPreflightGateJanusSurfaceAllowed ||
		gate.FinalGateObservationBoundaryPreflightGateCoocLearningAllowed ||
		gate.FinalGateObservationBoundaryPreflightGateDeltaHarvestAllowed ||
		gate.FinalGateObservationBoundaryPreflightGateBodyMutationAllowed ||
		!gate.FinalGateObservationBoundaryPreflightGatePreStateHashRequired ||
		!gate.FinalGateObservationBoundaryPreflightGatePostStateHashRequired ||
		!gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		!gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightConsumed ||
		!gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightRequired ||
		!gate.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate ||
		gate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema ||
		gate.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run" ||
		gate.SourceTarget != "live_route_admission_next_step" ||
		gate.SourceReport != preflightPath ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID != sourcePreflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID ||
		gate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausal != sourcePreflight.CausalID ||
		gate.SourceAdmissionFinalGateObservationBoundaryPreflightHash != sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightHash ||
		gate.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackHash != sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash ||
		gate.SourceAdmissionFinalGateObservationBoundaryPreflightReceiptShape != sourcePreflight.ReceiptShape ||
		gate.SourceAdmissionFinalGateObservationBoundaryPreflightAction != sourcePreflight.AdmissionFinalGateObservationBoundaryPreflightAction ||
		!gate.SourceAdmissionFinalGateObservationBoundaryPreflightDryRunOnly ||
		!gate.SourceAdmissionFinalGateObservationBoundaryPreflightBoundaryVerified ||
		!gate.SourceAdmissionFinalGateObservationBoundaryPreflightObservationVerified ||
		!gate.SourceAdmissionFinalGateObservationBoundaryPreflightReadBackVerified ||
		gate.SourceAdmissionFinalGateObservationBoundaryPreflightReady ||
		gate.SourceFinalGateObservationBoundaryPreflightKind != sourcePreflight.FinalGateObservationBoundaryPreflightKind ||
		gate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausalID(gate) ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateHash(gate) ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash(gate) ||
		gate.AdmissionFinalGateObservationBoundaryPreflightGateHash == gate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash ||
		gate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID(gate) ||
		gate.LedgerReady ||
		gate.LedgerAppendAllowed ||
		gate.WriteAllowed ||
		gate.AdmissionAllowed ||
		gate.LiveAdmissionEnabled ||
		gate.MutatesState ||
		gate.BodyMutationAllowed ||
		gate.AuthorityGranted ||
		gate.BodyTarget != "none" ||
		!gate.Passed ||
		gate.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate checked from blocked preflight; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate lost contract: %+v", gate)
	}

	notReadyPath := filepath.Join(dir, "not_ready_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{notReadyPath, filepath.Join(dir, "not_ready_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_ready not ready",
	)

	openedPreflightPath := filepath.Join(dir, "opened_preflight.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, openedPreflightPath)
	writeWeightedReadinessFixture(t, openedPreflightPath, stringsReplaceFirst(readText(t, openedPreflightPath), `"admission_final_gate_observation_boundary_preflight_ready": false`, `"admission_final_gate_observation_boundary_preflight_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{openedPreflightPath, filepath.Join(dir, "opened_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight opened admission_final_gate_observation_boundary_preflight_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_boundary_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-`, `"admission_final_gate_observation_boundary_preflight_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{badHashPath, filepath.Join(dir, "bad_hash_gate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight preflight_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGate([]string{preflightPath, filepath.Join(dir, "missing", "gate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate write failure, got %v", err)
	}
}
