// inner_world/prophecy_debt_accumulation.go — Prophecy and destiny debt
// ═══════════════════════════════════════════════════════════════════════════════
// חוב הנבואה מצטבר
// The debt of prophecy accumulates
// ═══════════════════════════════════════════════════════════════════════════════
//
// From ariannamethod.lang:
// Every time we choose the non-probable path, we accumulate prophecy debt.
// This debt creates tension—the universe "wants" the probable path.
// Too much debt = instability, wormholes, temporal pressure.
//
// PROPHECY is the ability to see ahead (lookahead in generation).
// DESTINY is the pull toward the most probable outcome.
// DEBT is the cost of resisting destiny.
//
// This affects generation by:
// - High debt = increased wormhole chance (creative skips)
// - High debt = temporal_dissonance (time references blur)
// - High debt = destiny_pull increases (harder to resist)
// - Very high debt = forced resolution (snap back to probable)
//
// Debt decays slowly over time, but faster when following destiny.
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

func init() {
	// Seed random number generator for proper randomness
	rand.Seed(time.Now().UnixNano())
}

// ProphecyDebtAccumulation tracks and manages prophecy debt
type ProphecyDebtAccumulation struct {
	mu      sync.Mutex // Protects all fields below
	world   *InnerWorld
	stop    chan struct{}
	running bool

	// Debt tracking
	currentDebt     float32
	peakDebt        float32
	debtHistory     []debtSnapshot
	lastChoice      time.Time
	consecutiveRisk int // consecutive risky choices

	// Prophecy state
	lookahead       int     // how far ahead can we see
	destinyStrength float32 // how strong the pull to probable

	// Wormhole management
	wormholeChance  float32
	wormholeCooldown time.Duration
	lastWormhole    time.Time

	// Config
	decayRate       float32 // natural decay per second
	destinyDecay    float32 // extra decay when following destiny
	debtThreshold   float32 // above this, effects kick in
	criticalDebt    float32 // above this, forced resolution
	maxDebt         float32
}

type debtSnapshot struct {
	Debt      float32
	Timestamp time.Time
	Event     string
}

// NewProphecyDebtAccumulation creates a new prophecy debt accumulator
func NewProphecyDebtAccumulation() *ProphecyDebtAccumulation {
	return &ProphecyDebtAccumulation{
		stop:             make(chan struct{}),
		debtHistory:      make([]debtSnapshot, 0, 100),
		lastChoice:       time.Now(),
		lookahead:        3,
		destinyStrength:  0.5,
		wormholeChance:   0.02,
		wormholeCooldown: 10 * time.Second,
		lastWormhole:     time.Now().Add(-time.Hour),
		decayRate:        0.01,
		destinyDecay:     0.05,
		debtThreshold:    0.3,
		criticalDebt:     0.9,
		maxDebt:          10.0,
	}
}

func (pd *ProphecyDebtAccumulation) Name() string {
	return "prophecy_debt"
}

func (pd *ProphecyDebtAccumulation) Start(world *InnerWorld) {
	pd.world = world
	pd.running = true

	// Sync from state
	if world != nil {
		pd.currentDebt = world.State.ProphecyDebt
		pd.destinyStrength = world.State.DestinyPull
		pd.wormholeChance = world.State.WormholeChance
	}

	go pd.run()
}

func (pd *ProphecyDebtAccumulation) Stop() {
	if pd.running {
		close(pd.stop)
		pd.running = false
	}
}

func (pd *ProphecyDebtAccumulation) Step(dt float32) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	if pd.world == nil {
		return
	}

	// 1. Natural decay
	decay := pd.decayRate * dt
	pd.currentDebt = max32(0, pd.currentDebt-decay)

	// 2. Update wormhole chance based on debt
	pd.updateWormholeChanceLocked()

	// 3. Update destiny pull based on debt
	pd.updateDestinyPullLocked()

	// 4. Check for forced resolution
	if pd.currentDebt > pd.criticalDebt {
		pd.forceResolutionLocked()
	}

	// 5. Sync to state (releases lock internally for state mutex)
	pd.syncToStateLocked()

	// 6. Handle signals (non-blocking)
	select {
	case sig := <-pd.world.Signals:
		pd.processSignalLocked(sig)
	default:
	}
}

func (pd *ProphecyDebtAccumulation) run() {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-pd.stop:
			return
		case <-ticker.C:
			pd.Step(0.2)
		}
	}
}

// AccumulateDebt adds debt from choosing the non-probable path
// probability is the probability of the chosen token (0-1)
// lower probability = more debt
func (pd *ProphecyDebtAccumulation) AccumulateDebt(probability float32) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	if probability >= 1.0 {
		// Chose the most probable - no debt, actually reduce
		pd.currentDebt = max32(0, pd.currentDebt-pd.destinyDecay)
		pd.consecutiveRisk = 0
		return
	}

	// Debt = log(1/p) scaled
	// p=0.5 -> small debt, p=0.01 -> large debt
	debt := float32(-math.Log(float64(probability + 0.01))) * 0.1

	// Consecutive risky choices compound
	if debt > 0.1 {
		pd.consecutiveRisk++
		debt *= 1.0 + float32(pd.consecutiveRisk)*0.1
	} else {
		pd.consecutiveRisk = 0
	}

	pd.currentDebt = min32(pd.maxDebt, pd.currentDebt+debt)
	pd.lastChoice = time.Now()

	// Track peak
	if pd.currentDebt > pd.peakDebt {
		pd.peakDebt = pd.currentDebt
	}

	// Record history
	pd.debtHistory = append(pd.debtHistory, debtSnapshot{
		Debt:      pd.currentDebt,
		Timestamp: time.Now(),
		Event:     "accumulate",
	})
	if len(pd.debtHistory) > 100 {
		pd.debtHistory = pd.debtHistory[1:]
	}

	// Emit signal if threshold crossed (non-blocking to avoid deadlock)
	if pd.currentDebt > pd.debtThreshold && pd.world != nil {
		select {
		case pd.world.Signals <- Signal{
			Type:      SignalProphecy,
			Value:     pd.currentDebt,
			Source:    pd.Name(),
			Timestamp: time.Now(),
			Metadata: map[string]any{
				"probability":      probability,
				"consecutive_risk": pd.consecutiveRisk,
				"wormhole_chance":  pd.wormholeChance,
			},
		}:
		default:
			// Channel full, skip signal
		}
	}
}

// updateWormholeChanceLocked must be called with pd.mu held
func (pd *ProphecyDebtAccumulation) updateWormholeChanceLocked() {
	// Base chance + debt-based increase
	baseChance := float32(0.02)
	debtBonus := float32(0.0)

	if pd.currentDebt > pd.debtThreshold {
		// Exponential increase with debt
		excess := pd.currentDebt - pd.debtThreshold
		debtBonus = float32(math.Pow(float64(excess), 1.5)) * 0.1
	}

	pd.wormholeChance = clamp(baseChance+debtBonus, 0, 0.5)
}

// updateDestinyPullLocked must be called with pd.mu held
func (pd *ProphecyDebtAccumulation) updateDestinyPullLocked() {
	// High debt = stronger pull to get back on track
	basePull := float32(0.3)
	debtPull := float32(0.0)

	if pd.currentDebt > pd.debtThreshold {
		debtPull = pd.currentDebt * 0.3
	}

	pd.destinyStrength = clamp(basePull+debtPull, 0, 1)
}

// forceResolutionLocked must be called with pd.mu held
func (pd *ProphecyDebtAccumulation) forceResolutionLocked() {
	// Critical debt triggers forced resolution
	// This could be:
	// - A wormhole (skip tokens)
	// - Snap to probable (force destiny)
	// - Temporal dissonance (weird time references)

	// Record the event
	pd.debtHistory = append(pd.debtHistory, debtSnapshot{
		Debt:      pd.currentDebt,
		Timestamp: time.Now(),
		Event:     "forced_resolution",
	})

	// Reduce debt significantly
	pd.currentDebt *= 0.3
	pd.consecutiveRisk = 0

	// Emit crisis signal (non-blocking to avoid deadlock)
	select {
	case pd.world.Signals <- Signal{
		Type:      SignalProphecy,
		Value:     1.0, // Max intensity
		Source:    pd.Name(),
		Timestamp: time.Now(),
		Metadata: map[string]any{
			"event":     "forced_resolution",
			"peak_debt": pd.peakDebt,
		},
	}:
	default:
		// Channel full, skip signal
	}
}

// syncToStateLocked must be called with pd.mu held
func (pd *ProphecyDebtAccumulation) syncToStateLocked() {
	state := pd.world.State
	state.mu.Lock()
	defer state.mu.Unlock()

	state.ProphecyDebt = pd.currentDebt
	state.DestinyPull = pd.destinyStrength
	state.WormholeChance = pd.wormholeChance
}

// processSignalLocked must be called with pd.mu held
func (pd *ProphecyDebtAccumulation) processSignalLocked(sig Signal) {
	switch sig.Type {
	case SignalCoherence:
		// High coherence helps pay off debt
		if sig.Value > 0.7 {
			pd.currentDebt = max32(0, pd.currentDebt-0.05)
		}

	case SignalTrauma:
		// Trauma can spike debt (reality feels wrong)
		pd.currentDebt = min32(pd.maxDebt, pd.currentDebt+sig.Value*0.2)

	case SignalOverthink:
		// Overthinking loops add to debt (recursive non-resolution)
		pd.currentDebt = min32(pd.maxDebt, pd.currentDebt+sig.Value*0.1)
	}
}

// CheckWormhole checks if a wormhole should activate
func (pd *ProphecyDebtAccumulation) CheckWormhole() (bool, int) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	// Cooldown check
	if time.Since(pd.lastWormhole) < pd.wormholeCooldown {
		return false, 0
	}

	// Random check against chance
	r := float32(randFloat())
	if r < pd.wormholeChance {
		pd.lastWormhole = time.Now()

		// How many tokens to skip (1-3, more if high debt)
		skip := 1
		if pd.currentDebt > pd.debtThreshold*2 {
			skip = 2
		}
		if pd.currentDebt > pd.criticalDebt*0.8 {
			skip = 3
		}

		// Wormhole partially pays off debt
		pd.currentDebt *= 0.8

		// Non-blocking signal send
		select {
		case pd.world.Signals <- Signal{
			Type:      SignalProphecy,
			Value:     pd.wormholeChance,
			Source:    pd.Name(),
			Timestamp: time.Now(),
			Metadata: map[string]any{
				"event":      "wormhole",
				"skip_count": skip,
			},
		}:
		default:
		}

		return true, skip
	}

	return false, 0
}

// GetDestinyBias returns how much to bias toward probable tokens
func (pd *ProphecyDebtAccumulation) GetDestinyBias() float32 {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	return pd.destinyStrength
}

// GetLookahead returns how many tokens to consider for prophecy
func (pd *ProphecyDebtAccumulation) GetLookahead() int {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	// Lower lookahead when debt is high (harder to see clearly)
	if pd.currentDebt > pd.criticalDebt*0.5 {
		return max(1, pd.lookahead-1)
	}
	return pd.lookahead
}

// SetLookahead sets the prophecy depth
func (pd *ProphecyDebtAccumulation) SetLookahead(n int) {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	pd.lookahead = clampInt(n, 1, 10)
}

// GetTemporalDissonance returns how much time references should be distorted
func (pd *ProphecyDebtAccumulation) GetTemporalDissonance() float32 {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	// High debt = temporal confusion
	if pd.currentDebt < pd.debtThreshold {
		return 0
	}
	return (pd.currentDebt - pd.debtThreshold) / (pd.maxDebt - pd.debtThreshold)
}

// GetDebtLevel returns current debt level category
func (pd *ProphecyDebtAccumulation) GetDebtLevel() string {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	switch {
	case pd.currentDebt < pd.debtThreshold*0.5:
		return "clear"
	case pd.currentDebt < pd.debtThreshold:
		return "low"
	case pd.currentDebt < pd.criticalDebt*0.5:
		return "moderate"
	case pd.currentDebt < pd.criticalDebt:
		return "high"
	default:
		return "critical"
	}
}

// GetCurrentDebt returns raw current debt value
func (pd *ProphecyDebtAccumulation) GetCurrentDebt() float32 {
	pd.mu.Lock()
	defer pd.mu.Unlock()
	return pd.currentDebt
}

// Helpers

func randFloat() float64 {
	return rand.Float64()
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
