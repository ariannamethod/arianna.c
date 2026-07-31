package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setBodyInventoryDefaultEnv(t *testing.T) {
	t.Helper()
	t.Setenv("A2A_JANUS_MODEL", "weights/arianna_v4_sft_f16.gguf")
	t.Setenv("A2A_RESONANCE_MODEL", "weights/arianna_resonance_v3_f16.gguf")
	t.Setenv("A2A_NANO_MODEL", "weights/nano_arianna_f16.gguf")
	t.Setenv("AM_BODY_INVENTORY_LOG", "")
	t.Setenv("AM_BODY_INVENTORY_REQUIRE_CORE", "")
}

func writeInventoryFile(t *testing.T, root, rel string, mode os.FileMode) {
	t.Helper()
	path := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte("present\n"), mode); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatalf("chmod %s: %v", path, err)
	}
}

func writeRequiredBodyInventoryFiles(t *testing.T, root string) {
	t.Helper()
	writeInventoryFile(t, root, "arianna", 0755)
	writeInventoryFile(t, root, "arianna_resonance", 0755)
	writeInventoryFile(t, root, "weights/arianna_v4_sft_f16.gguf", 0644)
	writeInventoryFile(t, root, "weights/arianna_resonance_v3_f16.gguf", 0644)
}

func writeOptionalBodyInventoryFiles(t *testing.T, root string) {
	t.Helper()
	writeInventoryFile(t, root, "nano-arianna", 0755)
	writeInventoryFile(t, root, "weights/nano_arianna_f16.gguf", 0644)
	writeInventoryFile(t, root, "doe_field", 0755)
	writeInventoryFile(t, root, "chorus-arianna", 0755)
	writeInventoryFile(t, root, "kk-cli", 0755)
	writeInventoryFile(t, root, "weights/nano.kk.db", 0644)
}

func inventoryMissingSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}

func sameStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func mustRouteAvailability(t *testing.T, receipt bodyInventoryReceipt, route string) bodyInventoryRouteAvailability {
	t.Helper()
	availability, ok := receipt.routeAvailability(route)
	if !ok {
		t.Fatalf("route availability missing for %s: %+v", route, receipt.RouteAvailability)
	}
	return availability
}

func TestInspectBodyInventoryReady(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()
	writeRequiredBodyInventoryFiles(t, root)
	writeOptionalBodyInventoryFiles(t, root)

	receipt := inspectBodyInventory(root)
	if receipt.Status != "ready" {
		t.Fatalf("status = %q, want ready", receipt.Status)
	}
	if !receipt.CoreReady || !receipt.OptionalReady || !receipt.LiveTrioAllowed {
		t.Fatalf("ready flags wrong: core=%t optional=%t live=%t", receipt.CoreReady, receipt.OptionalReady, receipt.LiveTrioAllowed)
	}
	if receipt.DegradedMode || !receipt.ContinueAllowed || receipt.MutatesState {
		t.Fatalf("contract flags wrong: degraded=%t continue=%t mutates=%t", receipt.DegradedMode, receipt.ContinueAllowed, receipt.MutatesState)
	}
	if len(receipt.RequiredMissing) != 0 || len(receipt.OptionalMissing) != 0 {
		t.Fatalf("unexpected missing required=%v optional=%v", receipt.RequiredMissing, receipt.OptionalMissing)
	}
	for _, route := range []string{"direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"} {
		if !receipt.routeAvailable(route) {
			t.Fatalf("route %s should be available in full inventory: %+v", route, mustRouteAvailability(t, receipt, route))
		}
	}
	direct := mustRouteAvailability(t, receipt, "direct")
	if !sameStrings(direct.RequiredOrgans, []string{"nano-weight"}) ||
		!sameStrings(direct.AnyOfOrgans, []string{"nano-binary", "doe-binary"}) {
		t.Fatalf("direct route organ contract changed: %+v", direct)
	}
}

func TestInspectBodyInventoryDegradedOnOptionalMissing(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()
	writeRequiredBodyInventoryFiles(t, root)

	receipt := inspectBodyInventory(root)
	if receipt.Status != "degraded" {
		t.Fatalf("status = %q, want degraded", receipt.Status)
	}
	if !receipt.CoreReady || receipt.OptionalReady || !receipt.LiveTrioAllowed {
		t.Fatalf("degraded flags wrong: core=%t optional=%t live=%t", receipt.CoreReady, receipt.OptionalReady, receipt.LiveTrioAllowed)
	}
	if !receipt.DegradedMode || !receipt.ContinueAllowed || receipt.MutatesState {
		t.Fatalf("contract flags wrong: degraded=%t continue=%t mutates=%t", receipt.DegradedMode, receipt.ContinueAllowed, receipt.MutatesState)
	}
	missing := inventoryMissingSet(receipt.OptionalMissing)
	for _, name := range []string{"nano-binary", "nano-weight", "doe-binary", "chorus-binary", "kk-binary", "kk-db"} {
		if !missing[name] {
			t.Fatalf("optional missing does not include %s: %v", name, receipt.OptionalMissing)
		}
	}
	for _, route := range []string{"direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"} {
		availability := mustRouteAvailability(t, receipt, route)
		if availability.Available {
			t.Fatalf("route %s should not be available without optional organs: %+v", route, availability)
		}
		if !strings.Contains(availability.Reason, "missing_route_organs:") {
			t.Fatalf("route %s missing route reason: %+v", route, availability)
		}
	}
}

func TestInspectBodyInventoryDoeCanCarryDirectRoute(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()
	writeRequiredBodyInventoryFiles(t, root)
	writeInventoryFile(t, root, "weights/nano_arianna_f16.gguf", 0644)
	writeInventoryFile(t, root, "doe_field", 0755)

	receipt := inspectBodyInventory(root)
	if receipt.Status != "degraded" {
		t.Fatalf("status = %q, want degraded", receipt.Status)
	}
	if !receipt.routeAvailable("direct") {
		t.Fatalf("direct route should be available through DOE + nano weight: %+v", mustRouteAvailability(t, receipt, "direct"))
	}
	for _, route := range []string{"chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"} {
		if receipt.routeAvailable(route) {
			t.Fatalf("route %s should need chorus-binary: %+v", route, mustRouteAvailability(t, receipt, route))
		}
	}
}

func TestRequireBodyInventoryLiveTrioAllowsDegradedOptionalMissing(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()
	writeRequiredBodyInventoryFiles(t, root)

	receipt := inspectBodyInventory(root)
	if err := requireBodyInventoryLiveTrio(receipt); err != nil {
		t.Fatalf("degraded optional inventory should still allow live trio: %v", err)
	}
	if !receipt.organPresent("janus-binary") || receipt.organPresent("nano-binary") {
		t.Fatalf("organ presence lookup wrong: %+v", receipt.Organs)
	}
}

func TestInspectBodyInventoryBlockedOnRequiredMissing(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()

	receipt := inspectBodyInventory(root)
	if receipt.Status != "blocked" {
		t.Fatalf("status = %q, want blocked", receipt.Status)
	}
	if receipt.CoreReady || receipt.LiveTrioAllowed || !receipt.ContinueAllowed {
		t.Fatalf("blocked flags wrong: core=%t live=%t continue=%t", receipt.CoreReady, receipt.LiveTrioAllowed, receipt.ContinueAllowed)
	}
	missing := inventoryMissingSet(receipt.RequiredMissing)
	for _, name := range []string{"janus-binary", "janus-weight", "resonance-binary", "resonance-weight"} {
		if !missing[name] {
			t.Fatalf("required missing does not include %s: %v", name, receipt.RequiredMissing)
		}
	}
}

func TestRequireBodyInventoryLiveTrioBlocksRequiredMissing(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	receipt := inspectBodyInventory(t.TempDir())

	err := requireBodyInventoryLiveTrio(receipt)
	if err == nil {
		t.Fatalf("required-missing inventory should block live trio")
	}
	if !strings.Contains(err.Error(), "janus-binary") || !strings.Contains(err.Error(), "resonance-weight") {
		t.Fatalf("blocked inventory error lost required organs: %v", err)
	}
}

func TestRunBodyInventorySmokeWritesReceiptAndOnlyRequiresCoreWhenAsked(t *testing.T) {
	setBodyInventoryDefaultEnv(t)
	root := t.TempDir()
	logPath := filepath.Join(t.TempDir(), "inventory.jsonl")
	t.Setenv("AM_BODY_INVENTORY_ROOT", root)
	t.Setenv("AM_BODY_INVENTORY_LOG", logPath)

	if err := runBodyInventorySmoke(); err != nil {
		t.Fatalf("smoke without require core failed: %v", err)
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read receipt log: %v", err)
	}
	logText := string(raw)
	if !strings.Contains(logText, bodyInventorySchema) || !strings.Contains(logText, `"status":"blocked"`) {
		t.Fatalf("receipt log missing schema/status: %s", logText)
	}

	t.Setenv("AM_BODY_INVENTORY_REQUIRE_CORE", "1")
	if err := runBodyInventorySmoke(); err == nil {
		t.Fatalf("smoke with require core succeeded despite missing required organs")
	}
}
