package main

import (
	"os"
	"sort"
	"strings"
	"testing"
)

const (
	admissionWeightedLSFGReceiverLegacyGoPrefix  = "admission_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"
	admissionWeightedLSFGReceiverCompactGoPrefix = "admission_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"

	admissionWeightedLSFGReceiverLegacyToolPrefix  = "admission_live_route_weighted_admission_resonance_graft_admission_final_gate_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"
	admissionWeightedLSFGReceiverCompactToolPrefix = "admission_live_route_weighted_lsfg_receiver_observation_boundary_preflight_gate_candidate_store_reader_proof_precondition_decision_promotion_switch_enable_gate"
)

func TestAdmissionWeightedLSFGReceiverNamespaceParity(t *testing.T) {
	requireNamespaceParity(t, ".", admissionWeightedLSFGReceiverLegacyGoPrefix, admissionWeightedLSFGReceiverCompactGoPrefix, ".go")
	requireNamespaceParity(t, "../tools", admissionWeightedLSFGReceiverLegacyToolPrefix, admissionWeightedLSFGReceiverCompactToolPrefix, ".sh")
}

func requireNamespaceParity(t *testing.T, dir, legacyPrefix, compactPrefix, suffixFilter string) {
	t.Helper()

	legacy := collectNamespaceSuffixes(t, dir, legacyPrefix, suffixFilter)
	compact := collectNamespaceSuffixes(t, dir, compactPrefix, suffixFilter)
	if len(legacy) == 0 {
		t.Fatalf("legacy namespace %q empty in %s", legacyPrefix, dir)
	}
	if len(compact) == 0 {
		t.Fatalf("compact namespace %q empty in %s", compactPrefix, dir)
	}

	var missingCompact []string
	for suffix := range legacy {
		if _, ok := compact[suffix]; !ok {
			missingCompact = append(missingCompact, compactPrefix+suffix)
		}
	}
	if len(missingCompact) > 0 {
		sort.Strings(missingCompact)
		t.Fatalf("compact namespace missing %d counterpart(s): %s", len(missingCompact), strings.Join(missingCompact, ", "))
	}

	var missingLegacy []string
	for suffix := range compact {
		if _, ok := legacy[suffix]; !ok {
			missingLegacy = append(missingLegacy, legacyPrefix+suffix)
		}
	}
	if len(missingLegacy) > 0 {
		sort.Strings(missingLegacy)
		t.Fatalf("legacy namespace missing %d counterpart(s): %s", len(missingLegacy), strings.Join(missingLegacy, ", "))
	}
}

func collectNamespaceSuffixes(t *testing.T, dir, prefix, suffixFilter string) map[string]struct{} {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}

	suffixes := make(map[string]struct{})
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffixFilter) {
			continue
		}
		suffixes[strings.TrimPrefix(name, prefix)] = struct{}{}
	}
	return suffixes
}
