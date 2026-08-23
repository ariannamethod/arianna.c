package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{"boundary.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{"boundary.json", "preflight.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{"  ", filepath.Join(dir, "preflight.json")}),
		"weighted admission resonance graft admission final gate observation boundary path missing",
	)

	boundaryPath := filepath.Join(dir, "boundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, boundaryPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{boundaryPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight output path missing",
	)

	preflightPath := filepath.Join(dir, "preflight.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{boundaryPath, preflightPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight rejected: %v", err)
	}
	raw, err := os.ReadFile(preflightPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight: %v", err)
	}
	var preflight admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReport
	if err := json.Unmarshal(raw, &preflight); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight: %v", err)
	}
	sourceRaw, err := os.ReadFile(boundaryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary: %v", err)
	}
	var sourceBoundary admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport
	if err := json.Unmarshal(sourceRaw, &sourceBoundary); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary: %v", err)
	}
	if preflight.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightSchema ||
		preflight.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_blocked_dry_run" ||
		preflight.Target != "live_route_admission_next_step" ||
		preflight.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight" ||
		preflight.TargetMode != "closed_preflight_guard_dry_run" ||
		preflight.Action != "check_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_dry_run" ||
		preflight.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" ||
		preflight.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" ||
		preflight.LedgerState != "blocked" ||
		preflight.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight" ||
		preflight.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_receipt" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightState != "blocked" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightAction != "check_blocked_final_gate_observation_boundary_preflight" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightTarget != "resonance" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary" ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightTargetMode != "closed_preflight_guard_dry_run" ||
		!preflight.AdmissionFinalGateObservationBoundaryPreflightDryRunOnly ||
		!preflight.AdmissionFinalGateObservationBoundaryPreflightBoundaryVerified ||
		!preflight.AdmissionFinalGateObservationBoundaryPreflightObservationVerified ||
		!preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackVerified ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightReady ||
		preflight.FinalGateObservationBoundaryPreflightKind != "blocked_final_gate_observation_boundary_preflight" ||
		preflight.FinalGateObservationBoundaryPreflightMode != "no_mutation_preflight" ||
		preflight.FinalGateObservationBoundaryPreflightStage != "post_observation_boundary_pre_live_admission" ||
		preflight.FinalGateObservationBoundaryPreflightRawDreamTextObserved ||
		preflight.FinalGateObservationBoundaryPreflightRawDreamTextForwarded ||
		preflight.FinalGateObservationBoundaryPreflightRawDreamTextAllowed ||
		preflight.FinalGateObservationBoundaryPreflightJanusSurfaceAllowed ||
		preflight.FinalGateObservationBoundaryPreflightCoocLearningAllowed ||
		preflight.FinalGateObservationBoundaryPreflightDeltaHarvestAllowed ||
		preflight.FinalGateObservationBoundaryPreflightBodyMutationAllowed ||
		!preflight.FinalGateObservationBoundaryPreflightPreStateHashRequired ||
		!preflight.FinalGateObservationBoundaryPreflightPostStateHashRequired ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryConsumed ||
		!preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryRequired ||
		!preflight.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflight ||
		preflight.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema ||
		preflight.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run" ||
		preflight.SourceTarget != "live_route_admission_next_step" ||
		preflight.SourceReport != boundaryPath ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID != sourceBoundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID ||
		preflight.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryCausal != sourceBoundary.CausalID ||
		preflight.SourceAdmissionFinalGateObservationBoundaryHash != sourceBoundary.AdmissionFinalGateObservationBoundaryHash ||
		preflight.SourceAdmissionFinalGateObservationBoundaryReadBackHash != sourceBoundary.AdmissionFinalGateObservationBoundaryReadBackHash ||
		preflight.SourceAdmissionFinalGateObservationBoundaryReceiptShape != sourceBoundary.ReceiptShape ||
		preflight.SourceAdmissionFinalGateObservationBoundaryAction != sourceBoundary.AdmissionFinalGateObservationBoundaryAction ||
		!preflight.SourceAdmissionFinalGateObservationBoundaryDryRunOnly ||
		!preflight.SourceAdmissionFinalGateObservationBoundaryObservationVerified ||
		!preflight.SourceAdmissionFinalGateObservationBoundaryReadBackVerified ||
		preflight.SourceAdmissionFinalGateObservationBoundaryReady ||
		preflight.SourceFinalGateObservationBoundaryKind != sourceBoundary.FinalGateObservationBoundaryKind ||
		preflight.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightCausalID(preflight) ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightHash(preflight) ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReadBackHash(preflight) ||
		preflight.AdmissionFinalGateObservationBoundaryPreflightHash == preflight.AdmissionFinalGateObservationBoundaryPreflightReadBackHash ||
		preflight.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightID(preflight) ||
		preflight.LedgerReady ||
		preflight.LedgerAppendAllowed ||
		preflight.WriteAllowed ||
		preflight.AdmissionAllowed ||
		preflight.LiveAdmissionEnabled ||
		preflight.MutatesState ||
		preflight.BodyMutationAllowed ||
		preflight.AuthorityGranted ||
		preflight.BodyTarget != "none" ||
		!preflight.Passed ||
		preflight.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight checked from blocked boundary; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight lost contract: %+v", preflight)
	}

	notReadyPath := filepath.Join(dir, "not_ready_boundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{notReadyPath, filepath.Join(dir, "not_ready_preflight.json")}),
		"weighted admission resonance graft admission final gate observation boundary weighted_admission_resonance_graft_admission_final_gate_observation_boundary_ready not ready",
	)

	openedBoundaryPath := filepath.Join(dir, "opened_boundary.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, openedBoundaryPath)
	writeWeightedReadinessFixture(t, openedBoundaryPath, stringsReplaceFirst(readText(t, openedBoundaryPath), `"admission_final_gate_observation_boundary_ready": false`, `"admission_final_gate_observation_boundary_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{openedBoundaryPath, filepath.Join(dir, "opened_preflight.json")}),
		"weighted admission resonance graft admission final gate observation boundary opened admission_final_gate_observation_boundary_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_boundary_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-`, `"admission_final_gate_observation_boundary_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{badHashPath, filepath.Join(dir, "bad_hash_preflight.json")}),
		"weighted admission resonance graft admission final gate observation boundary boundary_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflight([]string{boundaryPath, filepath.Join(dir, "missing", "preflight.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight write failure, got %v", err)
	}
}
