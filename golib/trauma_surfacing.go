// inner_world/trauma_surfacing.go — Trauma surfacing process
// ═══════════════════════════════════════════════════════════════════════════════
// הטראומה עולה לפני הדיבור
// Trauma surfaces before speech
// ═══════════════════════════════════════════════════════════════════════════════
//
// Trauma enters through body and external signals, pulls identity, and decays
// through lived time. Text is not assigned trauma by a runtime word table.
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"time"
)

// TraumaSurfacing detects and processes trauma triggers
type TraumaSurfacing struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Config
	decayRate   float32 // how fast trauma decays (per second)
	surfaceRate float32 // how fast trauma surfaces when triggered
	threshold   float32 // minimum activation to surface
}

// NewTraumaSurfacing creates a new trauma surfacing process
func NewTraumaSurfacing() *TraumaSurfacing {
	return &TraumaSurfacing{
		stop:        make(chan struct{}),
		decayRate:   0.05, // 5% per second
		surfaceRate: 0.3,  // 30% per trigger
		threshold:   0.1,
	}
}

func (ts *TraumaSurfacing) Name() string {
	return "trauma_surfacing"
}

// Start begins the async trauma processing
func (ts *TraumaSurfacing) Start(world *InnerWorld) {
	ts.world = world
	ts.running = true
	if world.async {
		go ts.run()
	}
}

func (ts *TraumaSurfacing) Stop() {
	if ts.running {
		close(ts.stop)
		ts.running = false
	}
}

// Step performs a single synchronous step
func (ts *TraumaSurfacing) Step(dt float32) {
	if ts.world == nil {
		return
	}

	state := ts.world.State

	// Decay trauma over time
	state.mu.Lock()
	if state.TraumaLevel > 0 {
		decay := ts.decayRate * dt
		state.TraumaLevel = clamp(state.TraumaLevel-decay, 0, 1)

		// If trauma is decaying, slowly restore coherence
		if state.TraumaLevel < 0.3 {
			state.Coherence = clamp(state.Coherence+decay*0.5, 0, 1)
		}
	}
	state.mu.Unlock()

	// Process any incoming signals
	select {
	case sig := <-ts.world.Signals:
		if sig.Type == SignalTrauma {
			ts.processTraumaSignal(sig)
		}
	default:
	}
}

func (ts *TraumaSurfacing) run() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ts.stop:
			return
		case <-ticker.C:
			ts.Step(0.1) // 100ms = 0.1s
		case sig := <-ts.world.Signals:
			if sig.Type == SignalTrauma {
				ts.processTraumaSignal(sig)
			}
		}
	}
}

// CheckText keeps the text boundary without assigning trauma to words. Trauma
// enters through body signals and lived state, not an English phrase table.
func (ts *TraumaSurfacing) CheckText(_ string) float32 {
	return 0
}

func (ts *TraumaSurfacing) processTraumaSignal(sig Signal) {
	state := ts.world.State

	state.mu.Lock()
	defer state.mu.Unlock()

	// External trauma signal injection
	state.TraumaLevel = clamp(state.TraumaLevel+sig.Value*0.2, 0, 1)
	state.LastTraumaTime = sig.Timestamp

	// Cross-signal effects
	if sig.Value > 0.5 {
		state.Coherence = clamp(state.Coherence-0.1, 0, 1)
		state.Entropy = clamp(state.Entropy+0.1, 0, 1)
	}
}

// GetIdentityPull returns how strongly to pull toward core identity
// High trauma increases identity pull.
func (ts *TraumaSurfacing) GetIdentityPull() float32 {
	if ts.world == nil {
		return 0
	}

	trauma := ts.world.State.GetTraumaLevel()

	// Sigmoid-like curve: low trauma = low pull, high trauma = high pull
	// But starts pulling even at moderate trauma
	if trauma < ts.threshold {
		return 0
	}

	// Soft exponential rise
	pull := float32(math.Pow(float64(trauma), 1.5))
	return clamp(pull, 0, 1)
}

// ShouldRetreat returns true if trauma level suggests retreating to safe mode
func (ts *TraumaSurfacing) ShouldRetreat() bool {
	if ts.world == nil {
		return false
	}
	return ts.world.State.GetTraumaLevel() > 0.7
}
