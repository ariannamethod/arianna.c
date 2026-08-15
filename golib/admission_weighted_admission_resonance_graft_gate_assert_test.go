package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert(t *testing.T) {
	dir := t.TempDir()

	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert(nil),
		"usage: --admission-live-route-weighted-admission-resonance-graft-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{"gate.json", "extra"}),
		"usage: --admission-live-route-weighted-admission-resonance-graft-gate-assert REPORT",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{"  "}),
		"weighted admission resonance graft gate path missing",
	)
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{filepath.Join(dir, "missing.json")}),
		"weighted admission resonance graft gate not written",
	)

	emptyPath := filepath.Join(dir, "empty.json")
	writeWeightedReadinessFixture(t, emptyPath, "")
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{emptyPath}),
		"weighted admission resonance graft gate not written",
	)

	invalidPath := filepath.Join(dir, "invalid.json")
	writeWeightedReadinessFixture(t, invalidPath, "{")
	err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{invalidPath})
	if err == nil || !strings.HasPrefix(err.Error(), "weighted admission resonance graft gate JSON invalid:") {
		t.Fatalf("expected invalid JSON error, got %v", err)
	}

	missingSchemaPath := filepath.Join(dir, "missing_schema.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, missingSchemaPath)
	gateText := readText(t, missingSchemaPath)
	writeWeightedReadinessFixture(t, missingSchemaPath, stringsReplaceFirst(gateText, `"schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v1",`, ""))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{missingSchemaPath}),
		"weighted admission resonance graft gate schema missing",
	)

	badSchemaPath := filepath.Join(dir, "bad_schema.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSchemaPath)
	writeWeightedReadinessFixture(t, badSchemaPath, stringsReplaceFirst(readText(t, badSchemaPath), `"schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v1"`, `"schema": "arianna.live_route_weighted_admission_resonance_graft_gate.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSchemaPath}),
		`weighted admission resonance graft gate schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_gate.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftGateSchema+`"`,
	)

	validPath := filepath.Join(dir, "valid.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, validPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{validPath}); err != nil {
		t.Fatalf("valid weighted admission resonance graft gate rejected: %v", err)
	}

	badStatusPath := filepath.Join(dir, "bad_status.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badStatusPath)
	writeWeightedReadinessFixture(t, badStatusPath, stringsReplaceFirst(readText(t, badStatusPath), `"status": "shadow_graft_gate_ready_dry_run"`, `"status": "open"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badStatusPath}),
		`weighted admission resonance graft gate status mismatch: got "open" want "shadow_graft_gate_ready_dry_run"`,
	)

	notReadyPath := filepath.Join(dir, "not_ready.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, notReadyPath)
	writeWeightedReadinessFixture(t, notReadyPath, stringsReplaceFirst(readText(t, notReadyPath), `"weighted_admission_resonance_graft_gate_ready": true`, `"weighted_admission_resonance_graft_gate_ready": false`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{notReadyPath}),
		"weighted admission resonance graft gate weighted_admission_resonance_graft_gate_ready not ready",
	)

	openedPath := filepath.Join(dir, "opened.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, openedPath)
	writeWeightedReadinessFixture(t, openedPath, stringsReplaceFirst(readText(t, openedPath), `"graft_allowed": false`, `"graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{openedPath}),
		"weighted admission resonance graft gate opened graft_allowed",
	)

	openedLivePath := filepath.Join(dir, "opened_live.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, openedLivePath)
	writeWeightedReadinessFixture(t, openedLivePath, stringsReplaceFirst(readText(t, openedLivePath), `"live_admission_enabled": false`, `"live_admission_enabled": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{openedLivePath}),
		"weighted admission resonance graft gate opened live_admission_enabled",
	)

	missingPathField := filepath.Join(dir, "missing_path.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, missingPathField)
	preflightReport := filepath.Join(dir, "preflight-"+filepath.Base(missingPathField))
	writeWeightedReadinessFixture(t, missingPathField, stringsReplaceFirst(readText(t, missingPathField), `"source_report": "`+preflightReport+`"`, `"source_report": " "`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{missingPathField}),
		"weighted admission resonance graft gate source_report missing",
	)

	badSourcePath := filepath.Join(dir, "bad_source.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSourcePath)
	writeWeightedReadinessFixture(t, badSourcePath, stringsReplaceFirst(readText(t, badSourcePath), `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v1"`, `"source_schema": "arianna.live_route_weighted_admission_resonance_graft_preflight.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSourcePath}),
		`weighted admission resonance graft gate source_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_preflight.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftPreflightSchema+`"`,
	)

	badSourcePreflightKindPath := filepath.Join(dir, "bad_source_preflight_kind.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSourcePreflightKindPath)
	writeWeightedReadinessFixture(t, badSourcePreflightKindPath, stringsReplaceFirst(readText(t, badSourcePreflightKindPath), `"source_preflight_kind": "shadow_graft_preflight"`, `"source_preflight_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSourcePreflightKindPath}),
		"weighted admission resonance graft gate source preflight shape mismatch",
	)

	badSourceGraftBoundarySchemaPath := filepath.Join(dir, "bad_source_graft_boundary_schema.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSourceGraftBoundarySchemaPath)
	writeWeightedReadinessFixture(t, badSourceGraftBoundarySchemaPath, stringsReplaceFirst(readText(t, badSourceGraftBoundarySchemaPath), `"source_graft_boundary_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v1"`, `"source_graft_boundary_schema": "arianna.live_route_weighted_admission_resonance_graft_boundary.v0"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSourceGraftBoundarySchemaPath}),
		`weighted admission resonance graft gate source_graft_boundary_schema mismatch: got "arianna.live_route_weighted_admission_resonance_graft_boundary.v0" want "`+admissionLiveRouteWeightedAdmissionResonanceGraftBoundarySchema+`"`,
	)

	badSourceBoundaryKindPath := filepath.Join(dir, "bad_source_boundary_kind.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSourceBoundaryKindPath)
	writeWeightedReadinessFixture(t, badSourceBoundaryKindPath, stringsReplaceFirst(readText(t, badSourceBoundaryKindPath), `"source_boundary_kind": "shadow_graft_boundary"`, `"source_boundary_kind": "live_graft"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSourceBoundaryKindPath}),
		"weighted admission resonance graft gate source boundary shape mismatch",
	)

	badGateHashPath := filepath.Join(dir, "bad_gate_hash.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badGateHashPath)
	writeWeightedReadinessFixture(t, badGateHashPath, stringsReplaceFirst(readText(t, badGateHashPath), `"gate_hash": "weighted-resonance-graft-gate-`, `"gate_hash": "weighted-resonance-graft-gate-bad`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badGateHashPath}),
		"weighted admission resonance graft gate gate_hash mismatch",
	)

	badSourceGuardPath := filepath.Join(dir, "bad_source_guard.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badSourceGuardPath)
	writeWeightedReadinessFixture(t, badSourceGuardPath, stringsReplaceFirst(readText(t, badSourceGuardPath), `"source_raw_dream_text_allowed": false`, `"source_raw_dream_text_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badSourceGuardPath}),
		"weighted admission resonance graft gate opened source_raw_dream_text_allowed",
	)

	badBoundaryGuardPath := filepath.Join(dir, "bad_boundary_guard.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badBoundaryGuardPath)
	writeWeightedReadinessFixture(t, badBoundaryGuardPath, stringsReplaceFirst(readText(t, badBoundaryGuardPath), `"source_boundary_graft_allowed": false`, `"source_boundary_graft_allowed": true`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badBoundaryGuardPath}),
		"weighted admission resonance graft gate opened source_boundary_graft_allowed",
	)

	badBodyTargetPath := filepath.Join(dir, "bad_body_target.json")
	writeWeightedAdmissionResonanceGraftGateFixture(t, badBodyTargetPath)
	writeWeightedReadinessFixture(t, badBodyTargetPath, stringsReplaceFirst(readText(t, badBodyTargetPath), `"body_target": "none"`, `"body_target": "live"`))
	requireBoundaryAssertError(t,
		runAdmissionLiveRouteWeightedAdmissionResonanceGraftGateAssert([]string{badBodyTargetPath}),
		`weighted admission resonance graft gate body_target mismatch: got "live" want "none"`,
	)
}

func writeWeightedAdmissionResonanceGraftGateFixture(t *testing.T, gatePath string) {
	t.Helper()
	dir := filepath.Dir(gatePath)
	preflightPath := filepath.Join(dir, "preflight-"+filepath.Base(gatePath))
	writeWeightedAdmissionResonanceGraftPreflightFixture(t, preflightPath)
	if err := runAdmissionLiveRouteWeightedAdmissionResonanceGraftGate([]string{preflightPath, gatePath}); err != nil {
		t.Fatalf("write weighted admission resonance graft gate fixture: %v", err)
	}
}
