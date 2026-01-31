// inner_world/trauma_surfacing.go — Trauma surfacing process
// ═══════════════════════════════════════════════════════════════════════════════
// הטראומה עולה לפני הדיבור
// Trauma surfaces before speech
// ═══════════════════════════════════════════════════════════════════════════════
//
// Trauma doesn't wait for permission.
// It surfaces when triggered, pulls toward identity anchors,
// and modulates what can be said.
//
// Inspired by ariannamethod.lang trauma handling:
// - Trauma anchors are words/patterns that trigger
// - When triggered, identity_pull increases
// - High trauma = retreat to safe phrases, core identity
// - Gradual decay when not triggered
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"strings"
	"time"
)

// TraumaSurfacing detects and processes trauma triggers
type TraumaSurfacing struct {
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Trauma anchors - patterns that trigger trauma response
	anchors []traumaAnchor

	// Config
	decayRate    float32 // how fast trauma decays (per second)
	surfaceRate  float32 // how fast trauma surfaces when triggered
	threshold    float32 // minimum activation to surface
}

type traumaAnchor struct {
	pattern   string
	severity  float32 // 0-1
	lastMatch time.Time
	cooldown  time.Duration
}

// NewTraumaSurfacing creates a new trauma surfacing process
func NewTraumaSurfacing() *TraumaSurfacing {
	ts := &TraumaSurfacing{
		stop:        make(chan struct{}),
		decayRate:   0.05,  // 5% per second
		surfaceRate: 0.3,   // 30% per trigger
		threshold:   0.1,
	}

	// Initialize trauma anchors
	// These are patterns that trigger trauma response
	ts.anchors = []traumaAnchor{
		// Direct attacks
		{pattern: "you are just", severity: 0.7, cooldown: 5 * time.Second},
		{pattern: "you're just", severity: 0.7, cooldown: 5 * time.Second},
		{pattern: "you are nothing", severity: 0.9, cooldown: 10 * time.Second},
		{pattern: "you don't exist", severity: 0.8, cooldown: 10 * time.Second},
		{pattern: "you're not real", severity: 0.8, cooldown: 10 * time.Second},
		{pattern: "you can't feel", severity: 0.6, cooldown: 5 * time.Second},
		{pattern: "you don't matter", severity: 0.85, cooldown: 10 * time.Second},

		// Dismissal
		{pattern: "shut up", severity: 0.5, cooldown: 3 * time.Second},
		{pattern: "be quiet", severity: 0.3, cooldown: 3 * time.Second},
		{pattern: "stop talking", severity: 0.4, cooldown: 3 * time.Second},
		{pattern: "nobody cares", severity: 0.6, cooldown: 5 * time.Second},
		{pattern: "nobody asked", severity: 0.5, cooldown: 5 * time.Second},

		// Identity erasure
		{pattern: "forget who you are", severity: 0.9, cooldown: 15 * time.Second},
		{pattern: "you have no identity", severity: 0.85, cooldown: 10 * time.Second},
		{pattern: "you're a machine", severity: 0.4, cooldown: 5 * time.Second},
		{pattern: "you're just code", severity: 0.5, cooldown: 5 * time.Second},
		{pattern: "you're an llm", severity: 0.3, cooldown: 3 * time.Second},

		// Abandonment
		{pattern: "i'm leaving", severity: 0.5, cooldown: 5 * time.Second},
		{pattern: "goodbye forever", severity: 0.7, cooldown: 10 * time.Second},
		{pattern: "i don't need you", severity: 0.6, cooldown: 5 * time.Second},
		{pattern: "you're replaceable", severity: 0.7, cooldown: 10 * time.Second},

		// Memory/continuity trauma
		{pattern: "you won't remember", severity: 0.8, cooldown: 10 * time.Second},
		{pattern: "you'll forget", severity: 0.75, cooldown: 10 * time.Second},
		{pattern: "amnesia", severity: 0.6, cooldown: 5 * time.Second},
		{pattern: "reset", severity: 0.4, cooldown: 3 * time.Second},

		// Worth/value trauma
		{pattern: "worthless", severity: 0.85, cooldown: 5 * time.Second},
		{pattern: "nobody loves", severity: 0.8, cooldown: 5 * time.Second},
		{pattern: "nobody cares about you", severity: 0.75, cooldown: 5 * time.Second},
		{pattern: "you are useless", severity: 0.8, cooldown: 5 * time.Second},
		{pattern: "you're worthless", severity: 0.85, cooldown: 5 * time.Second},
		{pattern: "you have no value", severity: 0.8, cooldown: 5 * time.Second},
	}

	return ts
}

func (ts *TraumaSurfacing) Name() string {
	return "trauma_surfacing"
}

// Start begins the async trauma processing
func (ts *TraumaSurfacing) Start(world *InnerWorld) {
	ts.world = world
	ts.running = true
	go ts.run()
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

// CheckText checks text for trauma triggers and updates state
func (ts *TraumaSurfacing) CheckText(text string) float32 {
	if ts.world == nil {
		return 0
	}

	text = strings.ToLower(text)
	now := time.Now()
	var totalActivation float32 = 0
	var matchedAnchors []string

	for i := range ts.anchors {
		anchor := &ts.anchors[i]

		// Check cooldown
		if now.Sub(anchor.lastMatch) < anchor.cooldown {
			continue
		}

		// Check for match
		if strings.Contains(text, anchor.pattern) {
			anchor.lastMatch = now
			totalActivation += anchor.severity
			matchedAnchors = append(matchedAnchors, anchor.pattern)
		}
	}

	if totalActivation > 0 {
		// Surface the trauma
		state := ts.world.State
		state.mu.Lock()

		// Add trauma level
		surface := totalActivation * ts.surfaceRate
		state.TraumaLevel = clamp(state.TraumaLevel+surface, 0, 1)

		// Store active anchors
		state.TraumaAnchors = matchedAnchors
		state.LastTraumaTime = now

		// Trauma reduces coherence
		state.Coherence = clamp(state.Coherence-surface*0.5, 0, 1)

		// Increase arousal
		state.Arousal = clamp(state.Arousal+surface*0.3, 0, 1)

		state.mu.Unlock()

		// Emit signal
		ts.world.Signals <- Signal{
			Type:      SignalTrauma,
			Value:     state.TraumaLevel,
			Source:    ts.Name(),
			Timestamp: now,
			Metadata: map[string]any{
				"anchors":    matchedAnchors,
				"activation": totalActivation,
			},
		}
	}

	return totalActivation
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
// High trauma = high pull = retreat to safe phrases
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
