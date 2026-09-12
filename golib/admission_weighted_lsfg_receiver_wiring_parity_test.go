package main

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

const (
	admissionWeightedLSFGReceiverGoFilePrefix   = "admission_weighted_lsfg_receiver_"
	admissionWeightedLSFGReceiverToolFilePrefix = "admission_live_route_weighted_lsfg_receiver_"
)

var admissionWeightedLSFGReceiverRunFuncRE = regexp.MustCompile(`func (runAdmissionLiveRouteWeightedAdmissionResonanceGraft[^(]+)\(`)

func TestAdmissionWeightedLSFGReceiverWiringParity(t *testing.T) {
	requireMetabolismDispatchForCompactLSFGReceiverRunFuncs(t)
	requireMakefileRecipesForCompactLSFGReceiverSmokes(t)
}

func requireMetabolismDispatchForCompactLSFGReceiverRunFuncs(t *testing.T) {
	t.Helper()

	metabolism := readTextFileForParity(t, "metabolism.go")
	runFuncs := collectCompactLSFGReceiverRunFuncs(t)

	var missing []string
	for _, fn := range runFuncs {
		if !strings.Contains(metabolism, fn+"(") {
			missing = append(missing, fn)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("metabolism dispatch missing %d compact LSFG receiver run func(s): %s", len(missing), strings.Join(missing, ", "))
	}
}

func requireMakefileRecipesForCompactLSFGReceiverSmokes(t *testing.T) {
	t.Helper()

	makefile := readTextFileForParity(t, "../Makefile")
	smokes := collectCompactLSFGReceiverSmokeScripts(t)

	var missing []string
	for _, script := range smokes {
		recipe := "bash tools/" + script
		if !strings.Contains(makefile, recipe) {
			missing = append(missing, script)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("Makefile missing %d compact LSFG receiver smoke recipe(s): %s", len(missing), strings.Join(missing, ", "))
	}
}

func collectCompactLSFGReceiverRunFuncs(t *testing.T) []string {
	t.Helper()

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read golib: %v", err)
	}

	funcs := make(map[string]struct{})
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, admissionWeightedLSFGReceiverGoFilePrefix) ||
			!strings.HasSuffix(name, ".go") ||
			strings.HasSuffix(name, "_test.go") {
			continue
		}

		text := readTextFileForParity(t, name)
		for _, match := range admissionWeightedLSFGReceiverRunFuncRE.FindAllStringSubmatch(text, -1) {
			funcs[match[1]] = struct{}{}
		}
	}

	if len(funcs) == 0 {
		t.Fatalf("no compact LSFG receiver run funcs found")
	}
	return sortedKeys(funcs)
}

func collectCompactLSFGReceiverSmokeScripts(t *testing.T) []string {
	t.Helper()

	entries, err := os.ReadDir("../tools")
	if err != nil {
		t.Fatalf("read tools: %v", err)
	}

	smokes := make(map[string]struct{})
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, admissionWeightedLSFGReceiverToolFilePrefix) {
			continue
		}
		if strings.HasSuffix(name, "_smoke.sh") || strings.HasSuffix(name, "_consumer_smoke.sh") {
			smokes[name] = struct{}{}
		}
	}

	if len(smokes) == 0 {
		t.Fatalf("no compact LSFG receiver smoke scripts found")
	}
	return sortedKeys(smokes)
}

func readTextFileForParity(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}

func sortedKeys(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for value := range values {
		keys = append(keys, value)
	}
	sort.Strings(keys)
	return keys
}
