package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{"observation.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{"observation.json", "boundary.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{"  ", filepath.Join(dir, "boundary.json")}),
		"weighted admission resonance graft admission final gate observation path missing",
	)

	observationPath := filepath.Join(dir, "observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, observationPath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{observationPath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary output path missing",
	)

	boundaryPath := filepath.Join(dir, "boundary.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{observationPath, boundaryPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary rejected: %v", err)
	}
	raw, err := os.ReadFile(boundaryPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary: %v", err)
	}
	var boundary admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReport
	if err := json.Unmarshal(raw, &boundary); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary: %v", err)
	}
	sourceRaw, err := os.ReadFile(observationPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation: %v", err)
	}
	var sourceObservation admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReport
	if err := json.Unmarshal(sourceRaw, &sourceObservation); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation: %v", err)
	}
	if boundary.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundarySchema ||
		boundary.Status != "shadow_graft_admission_final_gate_observation_boundary_declared_dry_run" ||
		boundary.Target != "live_route_admission_next_step" ||
		boundary.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary" ||
		boundary.TargetMode != "receipt_only_closed_dry_run" ||
		boundary.Action != "declare_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_dry_run" ||
		boundary.WriterAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.LedgerState != "blocked" ||
		boundary.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary" ||
		boundary.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_receipt" ||
		boundary.AdmissionFinalGateObservationBoundaryState != "declared" ||
		boundary.AdmissionFinalGateObservationBoundaryAction != "declare_blocked_final_gate_observation_boundary" ||
		boundary.AdmissionFinalGateObservationBoundaryTarget != "resonance" ||
		boundary.AdmissionFinalGateObservationBoundaryTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation" ||
		boundary.AdmissionFinalGateObservationBoundaryTargetMode != "receipt_only_closed_dry_run" ||
		!boundary.AdmissionFinalGateObservationBoundaryDryRunOnly ||
		!boundary.AdmissionFinalGateObservationBoundaryObservationVerified ||
		!boundary.AdmissionFinalGateObservationBoundaryReadBackVerified ||
		boundary.AdmissionFinalGateObservationBoundaryReady ||
		boundary.FinalGateObservationBoundaryKind != "blocked_final_gate_observation_boundary" ||
		boundary.FinalGateObservationBoundaryMode != "no_mutation_closed_boundary_receipt" ||
		boundary.FinalGateObservationBoundaryStage != "post_observation_pre_live_admission" ||
		boundary.FinalGateObservationBoundaryRawDreamTextObserved ||
		boundary.FinalGateObservationBoundaryRawDreamTextForwarded ||
		boundary.FinalGateObservationBoundaryRawDreamTextAllowed ||
		boundary.FinalGateObservationBoundaryJanusSurfaceAllowed ||
		boundary.FinalGateObservationBoundaryCoocLearningAllowed ||
		boundary.FinalGateObservationBoundaryDeltaHarvestAllowed ||
		boundary.FinalGateObservationBoundaryBodyMutationAllowed ||
		!boundary.FinalGateObservationBoundaryPreStateHashRequired ||
		!boundary.FinalGateObservationBoundaryPostStateHashRequired ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationConsumed ||
		!boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationRequired ||
		!boundary.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundary ||
		boundary.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationSchema ||
		boundary.SourceStatus != "shadow_graft_admission_final_gate_observation_recorded_dry_run" ||
		boundary.SourceTarget != "live_route_admission_next_step" ||
		boundary.SourceReport != observationPath ||
		boundary.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationID != sourceObservation.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationID ||
		boundary.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationCausal != sourceObservation.CausalID ||
		boundary.SourceAdmissionFinalGateObservationAppendHash != sourceObservation.AdmissionFinalGateObservationAppendHash ||
		boundary.SourceAdmissionFinalGateObservationReadBackHash != sourceObservation.AdmissionFinalGateObservationReadBackHash ||
		boundary.SourceAdmissionFinalGateObservationReceiptShape != sourceObservation.ReceiptShape ||
		boundary.SourceAdmissionFinalGateObservationAction != sourceObservation.AdmissionFinalGateObservationAction ||
		!boundary.SourceAdmissionFinalGateObservationDryRunOnly ||
		!boundary.SourceAdmissionFinalGateObservationAppendOnly ||
		!boundary.SourceAdmissionFinalGateObservationReadBack ||
		!boundary.SourceAdmissionFinalGateObservationReceiptVerified ||
		boundary.SourceAdmissionFinalGateObservationReady ||
		boundary.SourceFinalGateObservationObserver != sourceObservation.FinalGateObservationObserver ||
		boundary.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryCausalID(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryHash(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReadBackHash(boundary) ||
		boundary.AdmissionFinalGateObservationBoundaryHash == boundary.AdmissionFinalGateObservationBoundaryReadBackHash ||
		boundary.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryID(boundary) ||
		boundary.LedgerReady ||
		boundary.LedgerAppendAllowed ||
		boundary.WriteAllowed ||
		boundary.AdmissionAllowed ||
		boundary.LiveAdmissionEnabled ||
		boundary.MutatesState ||
		boundary.BodyMutationAllowed ||
		boundary.AuthorityGranted ||
		boundary.BodyTarget != "none" ||
		!boundary.Passed ||
		boundary.Reason != "weighted resonance shadow graft admission final gate observation boundary declared from recorded observation; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary lost contract: %+v", boundary)
	}

	notReadyPath := filepath.Join(dir, "not_ready_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{notReadyPath, filepath.Join(dir, "not_ready_boundary.json")}),
		"weighted admission resonance graft admission final gate observation weighted_admission_resonance_graft_admission_final_gate_observation_ready not ready",
	)

	openedObservationPath := filepath.Join(dir, "opened_observation.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, openedObservationPath)
	writeWeightedReadinessFixture(t, openedObservationPath, stringsReplaceFirst(readText(t, openedObservationPath), `"admission_final_gate_observation_ready": false`, `"admission_final_gate_observation_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{openedObservationPath, filepath.Join(dir, "opened_boundary.json")}),
		"weighted admission resonance graft admission final gate observation opened admission_final_gate_observation_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-append-`, `"admission_final_gate_observation_append_hash": "weighted-resonance-graft-admission-final-gate-observation-append-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{badHashPath, filepath.Join(dir, "bad_hash_boundary.json")}),
		"weighted admission resonance graft admission final gate observation append_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundary([]string{observationPath, filepath.Join(dir, "missing", "boundary.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary write failure, got %v", err)
	}
}
