// race_test.go — Race condition tests for inner_world
//
// Run with: go test -race ./inner_world/...
//
// These tests verify thread safety fixes for ProphecyDebtAccumulation,
// InnerWorld start/stop, and state access.

package main

import (
	"sync"
	"testing"
	"time"
)

// TestProphecyDebtConcurrentAccess tests concurrent access to ProphecyDebtAccumulation
func TestProphecyDebtConcurrentAccess(t *testing.T) {
	pd := NewProphecyDebtAccumulation()
	pd.world = NewInnerWorld()

	var wg sync.WaitGroup
	const numGoroutines = 10
	const numIterations = 100

	// Start multiple goroutines accessing prophecy debt
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				// Read operations
				_ = pd.GetCurrentDebt()
				_ = pd.GetDestinyBias()
				_ = pd.GetLookahead()
				_ = pd.GetDebtLevel()
				_ = pd.GetTemporalDissonance()

				// Write operations
				pd.AccumulateDebt(0.5)
				pd.SetLookahead(3)

				// Step (both read and write)
				pd.Step(0.016) // ~60fps
			}
		}()
	}

	wg.Wait()
	// If we get here without race detector triggering, test passes
}

// TestProphecyDebtCheckWormholeConcurrent tests concurrent wormhole checks
func TestProphecyDebtCheckWormholeConcurrent(t *testing.T) {
	pd := NewProphecyDebtAccumulation()
	pd.world = NewInnerWorld()

	var wg sync.WaitGroup
	const numGoroutines = 5
	const numIterations = 50

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				pd.CheckWormhole()
				pd.AccumulateDebt(0.3)
				time.Sleep(time.Microsecond)
			}
		}()
	}

	wg.Wait()
}

// TestInnerWorldStartStop tests start/stop cycle without race conditions
func TestInnerWorldStartStop(t *testing.T) {
	iw := NewInnerWorld()

	// Start and stop multiple times
	for i := 0; i < 5; i++ {
		iw.Start()
		time.Sleep(10 * time.Millisecond)
		iw.Stop()
	}
}

// TestInnerWorldConcurrentStartStop tests concurrent start/stop
func TestInnerWorldConcurrentStartStop(t *testing.T) {
	iw := NewInnerWorld()

	var wg sync.WaitGroup
	const numGoroutines = 3

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				iw.Start()
				time.Sleep(time.Millisecond)
				iw.Stop()
				time.Sleep(time.Millisecond)
			}
		}()
	}

	wg.Wait()
}

// TestInnerStateAccess tests concurrent access to InnerState
func TestInnerStateAccess(t *testing.T) {
	state := NewInnerState()

	var wg sync.WaitGroup
	const numGoroutines = 10
	const numIterations = 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				// Read operations
				_ = state.GetArousal()
				_ = state.GetValence()
				_ = state.GetEntropy()
				_ = state.GetCoherence()
				_ = state.GetTraumaLevel()
				_ = state.GetProphecyDebt()

				// Write operations
				state.SetArousal(0.5)
				state.SetValence(0.0)
				state.SetEntropy(0.3)
				state.SetCoherence(0.8)
				state.SetTraumaLevel(0.1)
				state.AddProphecyDebt(0.01)
			}
		}()
	}

	wg.Wait()
}

// TestSignalChannelConcurrent tests concurrent signal sending
func TestSignalChannelConcurrent(t *testing.T) {
	iw := NewInnerWorld()
	iw.Start()
	defer iw.Stop()

	var wg sync.WaitGroup
	const numGoroutines = 5
	const numSignals = 20

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numSignals; j++ {
				select {
				case iw.Signals <- Signal{
					Type:      SignalType(id % 10),
					Value:     float32(j) / float32(numSignals),
					Source:    "test",
					Timestamp: time.Now(),
				}:
				default:
					// Channel full, skip
				}
				time.Sleep(time.Microsecond)
			}
		}(i)
	}

	wg.Wait()
}

// TestMin32Max32 tests the helper functions are properly available
func TestMin32Max32(t *testing.T) {
	if min32(1.0, 2.0) != 1.0 {
		t.Error("min32(1, 2) should be 1")
	}
	if min32(2.0, 1.0) != 1.0 {
		t.Error("min32(2, 1) should be 1")
	}
	if max32(1.0, 2.0) != 2.0 {
		t.Error("max32(1, 2) should be 2")
	}
	if max32(2.0, 1.0) != 2.0 {
		t.Error("max32(2, 1) should be 2")
	}
}

// TestRandFloat tests that randFloat produces varied values
func TestRandFloat(t *testing.T) {
	values := make(map[float64]bool)
	for i := 0; i < 100; i++ {
		values[randFloat()] = true
	}

	// Should have many unique values (old implementation only had ~1000 max)
	if len(values) < 50 {
		t.Errorf("randFloat should produce varied values, got only %d unique in 100 calls", len(values))
	}
}
