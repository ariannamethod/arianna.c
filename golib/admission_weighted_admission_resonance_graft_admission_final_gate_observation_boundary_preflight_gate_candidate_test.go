package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{"gate.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{"gate.json", "candidate.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{"  ", filepath.Join(dir, "candidate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate path missing",
	)

	gatePath := filepath.Join(dir, "gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, gatePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{gatePath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate output path missing",
	)

	candidatePath := filepath.Join(dir, "candidate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{gatePath, candidatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate rejected: %v", err)
	}
	raw, err := os.ReadFile(candidatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate: %v", err)
	}
	var candidate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReport
	if err := json.Unmarshal(raw, &candidate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate: %v", err)
	}
	sourceRaw, err := os.ReadFile(gatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate: %v", err)
	}
	var sourceGate admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateReport
	if err := json.Unmarshal(sourceRaw, &sourceGate); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate: %v", err)
	}
	if candidate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateSchema ||
		candidate.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_blocked_dry_run" ||
		candidate.Target != "live_route_admission_next_step" ||
		candidate.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.TargetMode != "closed_preflight_gate_candidate_dry_run" ||
		candidate.Action != "draft_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_dry_run" ||
		candidate.WriterAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.RollbackAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.LedgerState != "blocked" ||
		candidate.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_receipt" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateState != "blocked" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateAction != "draft_blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTarget != "resonance" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate" ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateTargetMode != "closed_preflight_gate_candidate_dry_run" ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateDryRunOnly ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateGateVerified ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidatePreflightVerified ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateBoundaryVerified ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateObservationVerified ||
		!candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackVerified ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReady ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateMode != "no_mutation_preflight_gate_candidate" ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateStage != "post_preflight_gate_pre_live_admission" ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextObserved ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextForwarded ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateRawDreamTextAllowed ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateJanusSurfaceAllowed ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateCoocLearningAllowed ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateDeltaHarvestAllowed ||
		candidate.FinalGateObservationBoundaryPreflightGateCandidateBodyMutationAllowed ||
		!candidate.FinalGateObservationBoundaryPreflightGateCandidatePreStateHashRequired ||
		!candidate.FinalGateObservationBoundaryPreflightGateCandidatePostStateHashRequired ||
		!candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReady ||
		!candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateConsumed ||
		!candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateRequired ||
		!candidate.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate ||
		candidate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateSchema ||
		candidate.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_blocked_dry_run" ||
		candidate.SourceTarget != "live_route_admission_next_step" ||
		candidate.SourceReport != gatePath ||
		candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != sourceGate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		candidate.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCausal != sourceGate.CausalID ||
		candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateHash != sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateHash ||
		candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackHash != sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateReadBackHash ||
		candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReceiptShape != sourceGate.ReceiptShape ||
		candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateAction != sourceGate.AdmissionFinalGateObservationBoundaryPreflightGateAction ||
		!candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly ||
		!candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGatePreflightVerified ||
		!candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateBoundaryVerified ||
		!candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateObservationVerified ||
		!candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReadBackVerified ||
		candidate.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		candidate.SourceFinalGateObservationBoundaryPreflightGateKind != sourceGate.FinalGateObservationBoundaryPreflightGateKind ||
		candidate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateCausalID(candidate) ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash(candidate) ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash(candidate) ||
		candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateHash == candidate.AdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash ||
		candidate.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID(candidate) ||
		candidate.LedgerReady ||
		candidate.LedgerAppendAllowed ||
		candidate.WriteAllowed ||
		candidate.AdmissionAllowed ||
		candidate.LiveAdmissionEnabled ||
		candidate.MutatesState ||
		candidate.BodyMutationAllowed ||
		candidate.AuthorityGranted ||
		candidate.BodyTarget != "none" ||
		!candidate.Passed ||
		candidate.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate drafted from blocked gate; live admission remains closed" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate lost contract: %+v", candidate)
	}

	notReadyPath := filepath.Join(dir, "not_ready_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{notReadyPath, filepath.Join(dir, "not_ready_candidate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_ready not ready",
	)

	openedGatePath := filepath.Join(dir, "opened_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, openedGatePath)
	writeWeightedReadinessFixture(t, openedGatePath, stringsReplaceFirst(readText(t, openedGatePath), `"admission_final_gate_observation_boundary_preflight_gate_ready": false`, `"admission_final_gate_observation_boundary_preflight_gate_ready": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{openedGatePath, filepath.Join(dir, "opened_candidate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate opened admission_final_gate_observation_boundary_preflight_gate_ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_gate.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"admission_final_gate_observation_boundary_preflight_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-`, `"admission_final_gate_observation_boundary_preflight_gate_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{badHashPath, filepath.Join(dir, "bad_hash_candidate.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate gate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidate([]string{gatePath, filepath.Join(dir, "missing", "candidate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate write failure, got %v", err)
	}
}
