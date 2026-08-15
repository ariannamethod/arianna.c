package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate RESONANCE_GRAFT_GATE_REPORT RESONANCE_GRAFT_CANDIDATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{"gate.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate RESONANCE_GRAFT_GATE_REPORT RESONANCE_GRAFT_CANDIDATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{"gate.json", "candidate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate RESONANCE_GRAFT_GATE_REPORT RESONANCE_GRAFT_CANDIDATE_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{"  ", filepath.Join(dir, "candidate.json")}),
		"weighted admission resonance graft gate path missing",
	)

	gatePath := filepath.Join(dir, "gate.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, gatePath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{gatePath, "  "}),
		"weighted admission resonance graft candidate output path missing",
	)

	candidatePath := filepath.Join(dir, "candidate.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{gatePath, candidatePath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate rejected: %v", err)
	}
	raw, err := os.ReadFile(candidatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate: %v", err)
	}
	var candidate admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReport
	if err := json.Unmarshal(raw, &candidate); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate: %v", err)
	}
	var gate admissionLiveRouteWeightedAdmissionResonanceGraftGateReport
	gateRaw, err := os.ReadFile(gatePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft gate: %v", err)
	}
	if err := json.Unmarshal(gateRaw, &gate); err != nil {
		t.Fatalf("decode weighted admission resonance graft gate: %v", err)
	}
	if candidate.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateSchema ||
		candidate.Status != "shadow_graft_candidate_ready_dry_run" ||
		candidate.Target != "resonance" ||
		candidate.TargetKind != "weighted_internal_world_shadow_graft_candidate" ||
		candidate.TargetMode != "receipt_only_closed_candidate_dry_run" ||
		candidate.Action != "draft_weighted_resonance_shadow_graft_candidate_dry_run" ||
		!candidate.WeightedAdmissionResonanceGraftCandidateReady ||
		!candidate.WeightedAdmissionResonanceGraftGateConsumed ||
		!candidate.WeightedAdmissionResonanceGraftGateRequired ||
		!candidate.NextStepBlockedWithoutResonanceGraftCandidate ||
		candidate.ReceiptShape != "weighted_resonance_shadow_graft_candidate_contract" ||
		candidate.CandidateKind != "shadow_graft_candidate" ||
		candidate.CandidateMode != "no_mutation_candidate" ||
		candidate.CandidateStage != "pre_live_graft_candidate" ||
		candidate.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateCausalID(candidate) ||
		candidate.CandidateHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateHash(candidate) ||
		candidate.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateReadBackHash(candidate) ||
		candidate.CandidateHash == candidate.ReadBackHash ||
		candidate.WeightedAdmissionResonanceGraftCandidateID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateID(candidate) ||
		!candidate.PreflightVerified ||
		!candidate.BoundaryVerified ||
		!candidate.ObservationVerified ||
		!candidate.ReceiverVerified ||
		!candidate.IntentVerified ||
		!candidate.FinalGateVerified ||
		!candidate.SealVerified ||
		!candidate.PermitVerified ||
		!candidate.AuthorityVerified ||
		!candidate.AdmissionRequired ||
		!candidate.ShadowOnly ||
		candidate.GraftAllowed ||
		!candidate.DryRunOnly ||
		!candidate.LiveReady ||
		candidate.RawDreamTextAllowed ||
		candidate.RawDreamTextObserved ||
		candidate.RawDreamTextForwarded ||
		candidate.JanusSurfaceAllowed ||
		candidate.CoocLearningAllowed ||
		candidate.DeltaHarvestAllowed ||
		candidate.BodyMutationAllowed ||
		!candidate.RollbackRequired ||
		candidate.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema ||
		candidate.SourceStatus != "shadow_graft_gate_ready_dry_run" ||
		candidate.SourceReport != gatePath ||
		candidate.SourceWeightedAdmissionResonanceGraftGateID != gate.WeightedAdmissionResonanceGraftGateID ||
		!candidate.SourceWeightedAdmissionResonanceGraftGateReady ||
		candidate.SourceWeightedAdmissionResonanceGraftGateCausal != gate.CausalID ||
		candidate.SourceWeightedAdmissionResonanceGraftGateHash != gate.GateHash ||
		candidate.SourceWeightedAdmissionResonanceGraftGateRead != gate.ReadBackHash ||
		candidate.SourceGateAction != "gate_weighted_resonance_shadow_graft_dry_run" ||
		candidate.SourceGateReceiptShape != "weighted_resonance_shadow_graft_gate_contract" ||
		candidate.SourceGateKind != "shadow_graft_gate" ||
		candidate.SourceGateMode != "no_mutation_gate" ||
		candidate.SourceGateStage != "pre_live_graft_gate" ||
		!candidate.SourceGateShadowOnly ||
		candidate.SourceGateGraftAllowed ||
		!candidate.SourceGateDryRunOnly ||
		!candidate.SourceGateLiveReady ||
		candidate.SourceGateRawDreamTextAllowed ||
		candidate.SourceGateRawDreamTextObserved ||
		candidate.SourceGateRawDreamTextForwarded ||
		candidate.SourceGateJanusSurfaceAllowed ||
		candidate.SourceGateCoocLearningAllowed ||
		candidate.SourceGateDeltaHarvestAllowed ||
		candidate.SourceGateBodyMutationAllowed ||
		!candidate.SourceGateRollbackRequired ||
		!candidate.SourceGateNextStepBlockedWithoutResonanceGraftGate ||
		candidate.SourceWeightedAdmissionResonanceGraftPreflightID != gate.SourceWeightedAdmissionResonanceGraftPreflightID ||
		candidate.SourceWeightedAdmissionResonanceGraftBoundaryID != gate.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		candidate.SourceWeightedAdmissionResonanceObservationID != gate.SourceWeightedAdmissionResonanceObservationID ||
		candidate.SourceWeightedAdmissionResonanceReceiverID != gate.SourceWeightedAdmissionResonanceReceiverID ||
		!candidate.BodySmokeWeighted ||
		!candidate.NanoDirectRunner ||
		!candidate.NanoDirectFinalGate ||
		!candidate.ResonanceGraftAdmissionProof ||
		!candidate.BoundaryReportFullChain ||
		candidate.SourceAuthorityGranted ||
		candidate.AuthorityGranted ||
		candidate.ContractsReady ||
		candidate.WriteAllowed ||
		candidate.AdmissionAllowed ||
		candidate.LiveAdmissionEnabled ||
		candidate.MutatesState ||
		candidate.BodyTarget != "none" ||
		!candidate.Passed ||
		candidate.Reason != "weighted resonance shadow graft candidate drafted without body mutation" {
		t.Fatalf("weighted admission resonance graft candidate lost contract: %+v", candidate)
	}

	openedPath := filepath.Join(dir, "opened_gate.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{openedPath, filepath.Join(dir, "opened_candidate.json")}),
		"weighted admission resonance graft gate opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_gate.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{badSchemaPath, filepath.Join(dir, "bad_schema_candidate.json")}),
		`weighted admission resonance graft gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema+`"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready_gate.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_gate_ready": true`, `"weighted_admission_resonance_graft_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{notReadyPath, filepath.Join(dir, "not_ready_candidate.json")}),
		"weighted admission resonance graft gate weighted_admission_resonance_graft_gate_ready not ready",
	)

	badHashPath := filepath.Join(dir, "bad_hash_gate.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"gate_hash": "weighted-resonance-graft-gate-`, `"gate_hash": "weighted-resonance-graft-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{badHashPath, filepath.Join(dir, "bad_hash_candidate.json")}),
		"weighted admission resonance graft gate gate_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidate([]string{gatePath, filepath.Join(dir, "missing", "candidate.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate write failed:") {
		t.Fatalf("expected weighted admission resonance graft candidate write failure, got %v", err)
	}
}
