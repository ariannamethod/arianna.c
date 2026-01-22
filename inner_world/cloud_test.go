// test_cloud_go.go — Test for Cloud 200K Go implementation
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	fmt.Println("=" + string(make([]byte, 58, 58)) + "=")
	fmt.Println("  CLOUD 200K Go Implementation Test")
	fmt.Println("=" + string(make([]byte, 58, 58)) + "=")
	fmt.Println()

	// Change to repo root
	execPath, _ := os.Executable()
	repoRoot := filepath.Dir(filepath.Dir(execPath))
	os.Chdir(repoRoot)

	passed := 0
	failed := 0

	// Test 1: Create Cloud
	fmt.Println("Test 1: Cloud creation...")
	cloud := NewCloud(42)
	if cloud != nil && cloud.ParamCount() > 0 {
		fmt.Printf("  PASS: Cloud created with %d params\n", cloud.ParamCount())
		passed++
	} else {
		fmt.Println("  FAIL: Cloud creation failed")
		failed++
	}

	// Test 2: Resonance computation
	fmt.Println("\nTest 2: Resonance computation...")
	resonances := ComputeResonance("I feel terrified and anxious")
	nonZero := 0
	for _, r := range resonances {
		if r > 0 {
			nonZero++
		}
	}
	if nonZero > 0 {
		fmt.Printf("  PASS: Found %d non-zero resonances\n", nonZero)
		passed++
	} else {
		fmt.Println("  FAIL: No resonances detected")
		failed++
	}

	// Test 3: Ping sync
	fmt.Println("\nTest 3: Sync ping...")
	response := cloud.Ping("I'm feeling anxious and scared")
	if response != nil && response.Primary != "" {
		fmt.Printf("  PASS: Primary=%s, Secondary=%s, Iterations=%d\n",
			response.Primary, response.Secondary, response.Iterations)
		passed++
	} else {
		fmt.Println("  FAIL: Ping returned nil or empty")
		failed++
	}

	// Test 4: Chamber activations
	fmt.Println("\nTest 4: Chamber activations...")
	hasActivations := false
	for name, val := range response.ChamberActivations {
		if val > 0 {
			fmt.Printf("  %s: %.3f\n", name, val)
			hasActivations = true
		}
	}
	if hasActivations {
		fmt.Println("  PASS: Chambers have activations")
		passed++
	} else {
		fmt.Println("  FAIL: No chamber activations")
		failed++
	}

	// Test 5: Temperature bias
	fmt.Println("\nTest 5: Temperature bias...")
	if response.TemperatureBias >= -0.2 && response.TemperatureBias <= 0.2 {
		fmt.Printf("  PASS: Temperature bias = %.3f (in range)\n", response.TemperatureBias)
		passed++
	} else {
		fmt.Printf("  FAIL: Temperature bias = %.3f (out of range)\n", response.TemperatureBias)
		failed++
	}

	// Test 6: Async ping
	fmt.Println("\nTest 6: Async ping...")
	cloud.Start()
	responseChan := cloud.PingAsync("warmth and love")
	asyncResponse := <-responseChan
	cloud.Stop()
	if asyncResponse != nil && asyncResponse.Primary != "" {
		fmt.Printf("  PASS: Async Primary=%s\n", asyncResponse.Primary)
		passed++
	} else {
		fmt.Println("  FAIL: Async ping failed")
		failed++
	}

	// Test 7: Different emotions
	fmt.Println("\nTest 7: Emotion detection...")
	testCases := []struct {
		text     string
		expected string
	}{
		{"I'm terrified", "FEAR"},
		{"I love you so much", "LOVE"},
		{"This makes me furious", "RAGE"},
		{"I feel empty inside", "VOID"},
		{"I'm curious about this", "FLOW"},
		{"I feel shame and guilt", "COMPLEX"},
	}

	emotionsPassed := 0
	for _, tc := range testCases {
		resp := cloud.Ping(tc.text)
		// Find dominant chamber
		maxChamber := ""
		maxVal := float32(0)
		for name, val := range resp.ChamberActivations {
			if val > maxVal {
				maxVal = val
				maxChamber = name
			}
		}
		if maxChamber == tc.expected {
			fmt.Printf("  PASS: '%s' -> %s\n", tc.text[:20], maxChamber)
			emotionsPassed++
		} else {
			fmt.Printf("  FAIL: '%s' -> %s (expected %s)\n", tc.text[:20], maxChamber, tc.expected)
		}
	}
	if emotionsPassed >= 4 { // Allow some flexibility
		fmt.Printf("  PASS: %d/6 emotions correctly detected\n", emotionsPassed)
		passed++
	} else {
		fmt.Printf("  FAIL: Only %d/6 emotions detected\n", emotionsPassed)
		failed++
	}

	// Summary
	fmt.Println()
	fmt.Println("=" + string(make([]byte, 58, 58)) + "=")
	fmt.Printf("  Results: %d passed, %d failed\n", passed, failed)
	fmt.Println("=" + string(make([]byte, 58, 58)) + "=")

	if failed > 0 {
		os.Exit(1)
	}
}
