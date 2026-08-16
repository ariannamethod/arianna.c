package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader RESONANCE_GRAFT_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{"store.json"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader RESONANCE_GRAFT_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{"store.json", "reader.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-candidate-store-reader RESONANCE_GRAFT_CANDIDATE_STORE_REPORT RESONANCE_GRAFT_CANDIDATE_STORE_READER_REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{"  ", filepath.Join(dir, "reader.json")}),
		"weighted admission resonance graft candidate store path missing",
	)

	storePath := filepath.Join(dir, "store.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, storePath)

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{storePath, "  "}),
		"weighted admission resonance graft candidate store reader output path missing",
	)

	readerPath := filepath.Join(dir, "reader.json")
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{storePath, readerPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft candidate store reader rejected: %v", err)
	}
	raw, err := os.ReadFile(readerPath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate store reader: %v", err)
	}
	var reader admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReport
	if err := json.Unmarshal(raw, &reader); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate store reader: %v", err)
	}
	storeRaw, err := os.ReadFile(storePath)
	if err != nil {
		t.Fatalf("read weighted admission resonance graft candidate store: %v", err)
	}
	var store admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReport
	if err := json.Unmarshal(storeRaw, &store); err != nil {
		t.Fatalf("decode weighted admission resonance graft candidate store: %v", err)
	}
	if reader.Schema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderSchema ||
		reader.Status != "shadow_graft_candidate_store_read_back_dry_run" ||
		reader.Target != "resonance" ||
		reader.TargetKind != "weighted_internal_world_shadow_graft_candidate_store_reader" ||
		reader.TargetMode != "read_only_replay_dry_run" ||
		reader.Action != "read_weighted_resonance_shadow_graft_candidate_store_dry_run" ||
		!reader.WeightedAdmissionResonanceGraftCandidateStoreReaderReady ||
		!reader.WeightedAdmissionResonanceGraftCandidateStoreConsumed ||
		!reader.WeightedAdmissionResonanceGraftCandidateStoreRequired ||
		!reader.NextStepBlockedWithoutResonanceGraftCandidateStoreReader ||
		reader.ReceiptShape != "weighted_resonance_shadow_graft_candidate_store_reader_receipt" ||
		reader.ReaderKind != "shadow_graft_candidate_store_reader" ||
		reader.ReaderMode != "read_only_replay" ||
		reader.ReaderStage != "pre_live_graft_candidate_store_reader" ||
		reader.CausalID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderCausalID(reader) ||
		reader.ReaderHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderHash(reader) ||
		reader.ReplayHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReplayHash(reader) ||
		reader.ReadBackHash != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderReadBackHash(reader) ||
		reader.ReaderHash == reader.ReadBackHash ||
		reader.WeightedAdmissionResonanceGraftCandidateStoreReaderID != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReaderID(reader) ||
		!reader.StoreVerified ||
		!reader.CandidateVerified ||
		!reader.GateVerified ||
		!reader.PreflightVerified ||
		!reader.BoundaryVerified ||
		!reader.ObservationVerified ||
		!reader.ReceiverVerified ||
		!reader.IntentVerified ||
		!reader.FinalGateVerified ||
		!reader.SealVerified ||
		!reader.PermitVerified ||
		!reader.AuthorityVerified ||
		!reader.StoreHashVerified ||
		!reader.StoreReadBackVerified ||
		!reader.AdmissionRequired ||
		!reader.ShadowOnly ||
		reader.GraftAllowed ||
		!reader.DryRunOnly ||
		!reader.LiveReady ||
		reader.RawDreamTextAllowed ||
		reader.RawDreamTextObserved ||
		reader.RawDreamTextForwarded ||
		reader.JanusSurfaceAllowed ||
		reader.CoocLearningAllowed ||
		reader.DeltaHarvestAllowed ||
		reader.BodyMutationAllowed ||
		!reader.RollbackRequired ||
		!reader.ReadOnly ||
		!reader.ReplayOnly ||
		!reader.SourceAppendOnly ||
		!reader.SourceReadBack ||
		!reader.SourceReceiptPersisted ||
		!reader.SourceReceiptVerified ||
		reader.SourceSchema != admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema ||
		reader.SourceStatus != "shadow_graft_candidate_stored_dry_run" ||
		reader.SourceTarget != "resonance" ||
		reader.SourceReport != storePath ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateStoreID != store.WeightedAdmissionResonanceGraftCandidateStoreID ||
		!reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReady ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateStoreCausalID != store.CausalID ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateStoreHash != store.StoreHash ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateStoreReadBackHash != store.ReadBackHash ||
		reader.SourceStoreAction != "store_weighted_resonance_shadow_graft_candidate_dry_run" ||
		reader.SourceStoreReceiptShape != "weighted_resonance_shadow_graft_candidate_store_receipt" ||
		reader.SourceStoreKind != "shadow_graft_candidate_store" ||
		reader.SourceStoreMode != "append_only_read_back_store" ||
		reader.SourceStoreStage != "pre_live_graft_candidate_store" ||
		!reader.SourceStoreAppendOnly ||
		!reader.SourceStoreReadBack ||
		!reader.SourceStoreReceiptPersisted ||
		!reader.SourceStoreReceiptVerified ||
		!reader.SourceStoreAdmissionRequired ||
		!reader.SourceStoreShadowOnly ||
		reader.SourceStoreGraftAllowed ||
		!reader.SourceStoreDryRunOnly ||
		!reader.SourceStoreLiveReady ||
		reader.SourceStoreRawDreamTextAllowed ||
		reader.SourceStoreRawDreamTextObserved ||
		reader.SourceStoreRawDreamTextForwarded ||
		reader.SourceStoreJanusSurfaceAllowed ||
		reader.SourceStoreCoocLearningAllowed ||
		reader.SourceStoreDeltaHarvestAllowed ||
		reader.SourceStoreBodyMutationAllowed ||
		!reader.SourceStoreRollbackRequired ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateID != store.SourceWeightedAdmissionResonanceGraftCandidateID ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateHash != store.SourceWeightedAdmissionResonanceGraftCandidateHash ||
		reader.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash != store.SourceWeightedAdmissionResonanceGraftCandidateReadBackHash ||
		reader.SourceWeightedAdmissionResonanceGraftGateID != store.SourceWeightedAdmissionResonanceGraftGateID ||
		reader.SourceWeightedAdmissionResonanceGraftPreflightID != store.SourceWeightedAdmissionResonanceGraftPreflightID ||
		reader.SourceWeightedAdmissionResonanceGraftBoundaryID != store.SourceWeightedAdmissionResonanceGraftBoundaryID ||
		reader.SourceWeightedAdmissionResonanceObservationID != store.SourceWeightedAdmissionResonanceObservationID ||
		reader.SourceWeightedAdmissionResonanceReceiverID != store.SourceWeightedAdmissionResonanceReceiverID ||
		!reader.BodySmokeWeighted ||
		!reader.NanoDirectRunner ||
		!reader.NanoDirectFinalGate ||
		!reader.ResonanceGraftAdmissionProof ||
		!reader.BoundaryReportFullChain ||
		reader.SourceAuthorityGranted ||
		reader.AuthorityGranted ||
		reader.ContractsReady ||
		reader.WriteAllowed ||
		reader.AdmissionAllowed ||
		reader.LiveAdmissionEnabled ||
		reader.MutatesState ||
		reader.BodyTarget != "none" ||
		!reader.Passed ||
		reader.Reason != "weighted resonance shadow graft candidate store read back without body mutation" {
		t.Fatalf("weighted admission resonance graft candidate store reader lost contract: %+v", reader)
	}

	openedPath := filepath.Join(dir, "opened_store.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{openedPath, filepath.Join(dir, "opened_reader.json")}),
		"weighted admission resonance graft candidate store opened graft_allowed",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema_store.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{badSchemaPath, filepath.Join(dir, "bad_schema_reader.json")}),
		`weighted admission resonance graft candidate store schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_candidate_store.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreSchema+`"`,
	)

	badHashPath := filepath.Join(dir, "bad_hash_store.json")
	writeWeightedAdmissionResonanceGraftCandidateStoreFixture(t, badHashPath)
	writeWeightedReadinessFixture(t, badHashPath, stringsReplaceFirst(readText(t, badHashPath), `"store_hash": "weighted-resonance-graft-candidate-store-`, `"store_hash": "weighted-resonance-graft-candidate-store-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{badHashPath, filepath.Join(dir, "bad_hash_reader.json")}),
		"weighted admission resonance graft candidate store store_hash mismatch",
	)

	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftCandidateStoreReader([]string{storePath, filepath.Join(dir, "missing", "reader.json")}); err == nil ||
		!strings.HasPrefix(err.Error(), "weighted admission resonance graft candidate store reader write failed:") {
		t.Fatalf("expected weighted admission resonance graft candidate store reader write failure, got %v", err)
	}
}
