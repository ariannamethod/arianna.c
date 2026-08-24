package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader(t *testing.T) {
	dir := t.TempDir()

	usage := "usage: --admission-live-route-weighted-admission-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-reader RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_ADMISSION_FINAL_GATE_OBSERVATION_BOUNDARY_PREFLIGHT_GATE_CANDIDATE_STORE_READER_REPORT"
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader(nil), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{"store.json"}), usage)
	requireBoundaryAssertError(t, runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{"store.json", "reader.json", "extra"}), usage)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{"  ", filepath.Join(dir, "reader.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store path missing",
	)

	storePath := filepath.Join(dir, "store.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreFixture(t, storePath)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{storePath, "  "}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader output path missing",
	)

	readerPath := filepath.Join(dir, "reader.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{storePath, readerPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader rejected: %v", err)
	}
	raw, err := os.ReadFile(readerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader: %v", err)
	}
	var reader admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReport
	if err := json.Unmarshal(raw, &reader); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader: %v", err)
	}
	storeRaw, err := os.ReadFile(storePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft admission final gate observation boundary preflight gate candidate store: %v", err)
	}
	var store admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReport
	if err := json.Unmarshal(storeRaw, &store); err != nil {
		t.Fatalf("decode weighted admission resonance graft admission final gate observation boundary preflight gate candidate store: %v", err)
	}
	if reader.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderSchema ||
		reader.Status != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_read_back_dry_run" ||
		reader.Target != "live_route_admission_next_step" ||
		reader.TargetKind != "weighted_internal_world_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		reader.TargetMode != "read_only_replay_dry_run" ||
		reader.Action != "read_weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_dry_run" ||
		reader.LedgerState != "blocked" ||
		reader.LedgerAction != "reject_blocked_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_ledger_append" ||
		reader.LedgerReady ||
		reader.LedgerAppendAllowed ||
		!reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReady ||
		!reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreConsumed ||
		!reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreRequired ||
		!reader.NextStepBlockedWithoutResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader ||
		reader.ReceiptShape != "weighted_resonance_shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_receipt" ||
		reader.ReaderKind != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader" ||
		reader.ReaderMode != "read_only_replay" ||
		reader.ReaderStage != "post_preflight_gate_candidate_store_pre_live_admission_reader" ||
		reader.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderCausalID(reader) ||
		reader.ReaderHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderHash(reader) ||
		reader.ReplayHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReplayHash(reader) ||
		reader.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderReadBackHash(reader) ||
		reader.ReaderHash == reader.ReadBackHash ||
		reader.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReaderID(reader) ||
		!reader.StoreVerified ||
		!reader.CandidateVerified ||
		!reader.GateVerified ||
		!reader.PreflightVerified ||
		!reader.BoundaryVerified ||
		!reader.ObservationVerified ||
		!reader.FinalGateVerified ||
		!reader.SealVerified ||
		!reader.PermitVerified ||
		!reader.AuthorityVerified ||
		!reader.StoreHashVerified ||
		!reader.StoreReadBackVerified ||
		!reader.AdmissionRequired ||
		!reader.ShadowOnly ||
		!reader.DryRunOnly ||
		!reader.LiveReady ||
		!reader.RollbackRequired ||
		!reader.ReadOnly ||
		!reader.ReplayOnly ||
		reader.RawDreamTextAllowed ||
		reader.RawDreamTextObserved ||
		reader.RawDreamTextForwarded ||
		reader.BodyMutationAllowed ||
		reader.AuthorityGranted ||
		reader.ContractsReady ||
		reader.WriteAllowed ||
		reader.AdmissionAllowed ||
		reader.LiveAdmissionEnabled ||
		reader.MutatesState ||
		reader.BodyTarget != "none" ||
		!reader.Passed ||
		reader.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreSchema ||
		reader.SourceStatus != "shadow_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_stored_dry_run" ||
		reader.SourceTarget != "live_route_admission_next_step" ||
		reader.SourceReport != storePath ||
		reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID != store.WeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreID ||
		reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreCausal != store.CausalID ||
		reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreHash != store.StoreHash ||
		reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReadBackHash != store.ReadBackHash ||
		reader.SourceStoreReceiptShape != store.ReceiptShape ||
		reader.SourceStoreKind != store.StoreKind ||
		reader.SourceStoreMode != store.StoreMode ||
		reader.SourceStoreStage != store.StoreStage ||
		!reader.SourceStoreAppendOnly ||
		!reader.SourceStoreReadBack ||
		!reader.SourceStoreReceiptPersisted ||
		!reader.SourceStoreReceiptVerified ||
		reader.SourceStoreLedgerReady ||
		reader.SourceStoreLedgerAppendAllowed ||
		reader.SourceStoreRawDreamTextAllowed ||
		reader.SourceStoreBodyMutationAllowed ||
		reader.SourceStoreAuthorityGranted ||
		reader.SourceStoreWriteAllowed ||
		reader.SourceStoreLiveAdmissionEnabled ||
		reader.SourceStoreMutatesState ||
		reader.SourceStoreBodyTarget != "none" ||
		reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID != store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateID ||
		reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash != store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateHash ||
		reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash != store.SourceAdmissionFinalGateObservationBoundaryPreflightGateCandidateReadBackHash ||
		reader.SourceCandidateState != "blocked" ||
		reader.SourceCandidateKind != "blocked_final_gate_observation_boundary_preflight_gate_candidate" ||
		reader.SourceCandidateOpened ||
		reader.SourceCandidateBodyMutationAllowed ||
		reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID != store.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateID ||
		reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateReady ||
		!reader.SourceAdmissionFinalGateObservationBoundaryPreflightGateDryRunOnly ||
		reader.SourceFinalGateObservationBoundaryPreflightGateBodyMutationAllowed ||
		!reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightReady ||
		!reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryReady ||
		!reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateObservationReady ||
		!reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReceiverReady ||
		!reader.SourceWeightedAdmissionResonanceGraftAdmissionFinalGateReady ||
		!reader.SourceWriterInventoryVerified ||
		!reader.SourceWriterPreflightVerified ||
		!reader.SourceAdmissionRequired ||
		!reader.SourceShadowOnly ||
		!reader.SourceDryRunOnly ||
		!reader.SourceReadOnly ||
		!reader.SourceReplayOnly ||
		reader.Reason != "weighted resonance shadow graft admission final gate observation boundary preflight gate candidate store read back without ledger append or body mutation" {
		t.Fatalf("weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader lost contract: %+v", reader)
	}

	notReadyPath := filepath.Join(dir, "not_ready_store.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready": true`, `"weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{notReadyPath, filepath.Join(dir, "not_ready_reader.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened_store.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"ledger_append_allowed": false`, `"ledger_append_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{openedPath, filepath.Join(dir, "opened_reader.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store opened ledger_append_allowed",
	)

	badHashPath := filepath.Join(dir, "bad_hash_store.json")
	writeWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"store_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-`, `"store_hash": "weighted-resonance-graft-admission-final-gate-observation-boundary-preflight-gate-candidate-store-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{badHashPath, filepath.Join(dir, "bad_hash_reader.json")}),
		"weighted admission resonance graft admission final gate observation boundary preflight gate candidate store store_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftAdmissionFinalGateObservationBoundaryPreflightGateCandidateStoreReader([]string{storePath, filepath.Join(dir, "missing", "reader.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader write failed:") {
		t.Fatalf("expected weighted admission resonance graft admission final gate observation boundary preflight gate candidate store reader write failure, got %v", err)
	}
}
