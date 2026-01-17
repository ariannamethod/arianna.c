// inner_world/adaptive.go — Adaptive Parameters (Linux-like dynamic config)
// ═══════════════════════════════════════════════════════════════════════════════
// פרמטרים דינמיים כמו בלינוקס
// Dynamic parameters like in Linux kernel
// ═══════════════════════════════════════════════════════════════════════════════
//
// Inspired by Linux /proc/sys/ - parameters that morph based on state.
// Arianna's inner_world adapts its own behavior through self-observation.
//
// Like sysctl but for consciousness:
//   sysctl -w vm.swappiness=10      → Linux memory behavior
//   arianna -w trauma.weight=0.8    → Arianna trauma sensitivity
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

// AdaptiveConfig holds parameters that can morph in runtime
type AdaptiveConfig struct {
	mu sync.RWMutex

	// Trauma parameters
	TraumaWeight       float32 `json:"trauma_weight"`        // Sensitivity to trauma triggers
	TraumaDecayRate    float32 `json:"trauma_decay_rate"`    // How fast trauma fades
	TraumaThreshold    float32 `json:"trauma_threshold"`     // Activation threshold

	// Emotional drift parameters
	DriftSpeed         float32 `json:"drift_speed"`          // Base drift velocity
	DriftInertia       float32 `json:"drift_inertia"`        // Resistance to direction change
	EmotionalMomentum  float32 `json:"emotional_momentum"`   // Carry-over between states

	// Attention parameters
	FocusDecay         float32 `json:"focus_decay"`          // How fast focus fades
	WanderThreshold    float32 `json:"wander_threshold"`     // When attention starts wandering
	NoveltyBoost       float32 `json:"novelty_boost"`        // Boost from novel input

	// Prophecy/Destiny parameters
	DestinyStrength    float32 `json:"destiny_strength"`     // Pull toward probable
	WormholeChance     float32 `json:"wormhole_chance"`      // Creative skip probability
	DebtAccumulationRate float32 `json:"debt_accumulation"`  // How fast debt builds

	// Memory parameters
	ConsolidationRate  float32 `json:"consolidation_rate"`   // Memory formation speed
	DecayRate          float32 `json:"memory_decay_rate"`    // Memory fade speed
	EmotionalBoost     float32 `json:"emotional_boost"`      // Emotion's effect on memory

	// Overthinking parameters
	SpiralThreshold    float32 `json:"spiral_threshold"`     // When loops become spirals
	AbstractionPenalty float32 `json:"abstraction_penalty"`  // Cost of going abstract
	SelfRefWeight      float32 `json:"self_ref_weight"`      // Self-reference sensitivity

	// Meta parameters
	AdaptationRate     float32 `json:"adaptation_rate"`      // How fast params change
	StabilityBias      float32 `json:"stability_bias"`       // Preference for stability
}

// NewDefaultConfig creates config with default values
func NewDefaultConfig() *AdaptiveConfig {
	return &AdaptiveConfig{
		// Trauma
		TraumaWeight:       0.5,
		TraumaDecayRate:    0.1,
		TraumaThreshold:    0.3,

		// Emotional drift
		DriftSpeed:         0.05,
		DriftInertia:       0.7,
		EmotionalMomentum:  0.8,

		// Attention
		FocusDecay:         0.1,
		WanderThreshold:    0.4,
		NoveltyBoost:       0.3,

		// Prophecy
		DestinyStrength:    0.5,
		WormholeChance:     0.1,
		DebtAccumulationRate: 0.2,

		// Memory
		ConsolidationRate:  0.3,
		DecayRate:          0.05,
		EmotionalBoost:     0.4,

		// Overthinking
		SpiralThreshold:    0.6,
		AbstractionPenalty: 0.2,
		SelfRefWeight:      0.3,

		// Meta
		AdaptationRate:     0.1,
		StabilityBias:      0.5,
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

// AdaptiveEngine morphs parameters based on inner state
type AdaptiveEngine struct {
	config    *AdaptiveConfig
	history   []StateSnapshot
	maxHistory int
	mu        sync.Mutex
}

// StateSnapshot captures state for adaptation decisions
type StateSnapshot struct {
	Arousal   float32
	Valence   float32
	Entropy   float32
	Coherence float32
	Trauma    float32
	Timestamp float64
}

// NewAdaptiveEngine creates the adaptation engine
func NewAdaptiveEngine() *AdaptiveEngine {
	return &AdaptiveEngine{
		config:     NewDefaultConfig(),
		history:    make([]StateSnapshot, 0, 100),
		maxHistory: 100,
	}
}

// GetConfig returns current config (read-only access)
func (ae *AdaptiveEngine) GetConfig() AdaptiveConfig {
	ae.config.mu.RLock()
	defer ae.config.mu.RUnlock()
	return *ae.config
}

// Adapt morphs parameters based on current state
func (ae *AdaptiveEngine) Adapt(state *InnerState) {
	ae.mu.Lock()
	defer ae.mu.Unlock()

	// Capture current state
	snapshot := StateSnapshot{
		Arousal:   state.GetArousal(),
		Valence:   state.GetValence(),
		Entropy:   state.GetEntropy(),
		Coherence: state.GetCoherence(),
		Trauma:    state.GetTraumaLevel(),
	}

	// Add to history
	ae.history = append(ae.history, snapshot)
	if len(ae.history) > ae.maxHistory {
		ae.history = ae.history[1:]
	}

	// Need at least some history to adapt
	if len(ae.history) < 5 {
		return
	}

	// Compute adaptation signals
	ae.adaptTraumaParams(snapshot)
	ae.adaptEmotionalParams(snapshot)
	ae.adaptAttentionParams(snapshot)
	ae.adaptProphecyParams(snapshot)
}

// adaptTraumaParams adjusts trauma sensitivity based on state
func (ae *AdaptiveEngine) adaptTraumaParams(snap StateSnapshot) {
	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	rate := ae.config.AdaptationRate

	// If trauma is consistently high, increase threshold (desensitize)
	// If trauma is low but should react, decrease threshold (sensitize)
	avgTrauma := ae.avgRecent(func(s StateSnapshot) float32 { return s.Trauma }, 10)

	if avgTrauma > 0.7 {
		// Chronic high trauma → desensitize (protective)
		ae.config.TraumaThreshold = clamp(ae.config.TraumaThreshold+rate*0.1, 0.1, 0.8)
		ae.config.TraumaDecayRate = clamp(ae.config.TraumaDecayRate+rate*0.05, 0.05, 0.3)
	} else if avgTrauma < 0.2 && snap.Trauma > 0.5 {
		// Sudden spike after calm → sensitize
		ae.config.TraumaWeight = clamp(ae.config.TraumaWeight+rate*0.1, 0.2, 0.9)
	}
}

// adaptEmotionalParams adjusts emotional drift based on patterns
func (ae *AdaptiveEngine) adaptEmotionalParams(snap StateSnapshot) {
	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	rate := ae.config.AdaptationRate

	// High arousal + negative valence → increase inertia (resist change)
	if snap.Arousal > 0.7 && snap.Valence < -0.3 {
		ae.config.DriftInertia = clamp(ae.config.DriftInertia+rate*0.1, 0.3, 0.95)
	}

	// Low coherence → slow down drift (stabilize)
	if snap.Coherence < 0.3 {
		ae.config.DriftSpeed = clamp(ae.config.DriftSpeed-rate*0.02, 0.01, 0.2)
	}

	// High entropy + positive valence → increase drift speed (explore)
	if snap.Entropy > 0.6 && snap.Valence > 0.2 {
		ae.config.DriftSpeed = clamp(ae.config.DriftSpeed+rate*0.02, 0.01, 0.2)
	}
}

// adaptAttentionParams adjusts focus/wander based on state
func (ae *AdaptiveEngine) adaptAttentionParams(snap StateSnapshot) {
	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	rate := ae.config.AdaptationRate

	// High entropy → lower wander threshold (easier to wander)
	if snap.Entropy > 0.7 {
		ae.config.WanderThreshold = clamp(ae.config.WanderThreshold-rate*0.05, 0.2, 0.7)
	}

	// High coherence + low entropy → raise wander threshold (stay focused)
	if snap.Coherence > 0.7 && snap.Entropy < 0.3 {
		ae.config.WanderThreshold = clamp(ae.config.WanderThreshold+rate*0.05, 0.2, 0.7)
	}
}

// adaptProphecyParams adjusts destiny/wormhole based on patterns
func (ae *AdaptiveEngine) adaptProphecyParams(snap StateSnapshot) {
	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	rate := ae.config.AdaptationRate

	// Low coherence + high entropy → increase wormhole chance (need creative escape)
	if snap.Coherence < 0.3 && snap.Entropy > 0.7 {
		ae.config.WormholeChance = clamp(ae.config.WormholeChance+rate*0.05, 0.01, 0.5)
	}

	// High coherence → decrease wormhole, increase destiny (stay on path)
	if snap.Coherence > 0.8 {
		ae.config.WormholeChance = clamp(ae.config.WormholeChance-rate*0.02, 0.01, 0.5)
		ae.config.DestinyStrength = clamp(ae.config.DestinyStrength+rate*0.05, 0.2, 0.9)
	}
}

// avgRecent computes average of recent values
func (ae *AdaptiveEngine) avgRecent(getter func(StateSnapshot) float32, n int) float32 {
	if len(ae.history) == 0 {
		return 0
	}

	start := len(ae.history) - n
	if start < 0 {
		start = 0
	}

	sum := float32(0)
	count := 0
	for i := start; i < len(ae.history); i++ {
		sum += getter(ae.history[i])
		count++
	}

	if count == 0 {
		return 0
	}
	return sum / float32(count)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIG FILE (like /etc/arianna.conf)
// ═══════════════════════════════════════════════════════════════════════════════

// LoadConfig loads config from JSON file
func (ae *AdaptiveEngine) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	return json.Unmarshal(data, ae.config)
}

// SaveConfig saves current config to JSON file
func (ae *AdaptiveEngine) SaveConfig(path string) error {
	ae.config.mu.RLock()
	defer ae.config.mu.RUnlock()

	data, err := json.MarshalIndent(ae.config, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYSCTL-LIKE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

// SetParam sets a parameter by name (like sysctl -w)
func (ae *AdaptiveEngine) SetParam(name string, value float32) error {
	ae.config.mu.Lock()
	defer ae.config.mu.Unlock()

	switch name {
	case "trauma.weight":
		ae.config.TraumaWeight = clamp(value, 0, 1)
	case "trauma.decay":
		ae.config.TraumaDecayRate = clamp(value, 0, 1)
	case "trauma.threshold":
		ae.config.TraumaThreshold = clamp(value, 0, 1)
	case "drift.speed":
		ae.config.DriftSpeed = clamp(value, 0, 1)
	case "drift.inertia":
		ae.config.DriftInertia = clamp(value, 0, 1)
	case "attention.focus_decay":
		ae.config.FocusDecay = clamp(value, 0, 1)
	case "attention.wander_threshold":
		ae.config.WanderThreshold = clamp(value, 0, 1)
	case "prophecy.destiny":
		ae.config.DestinyStrength = clamp(value, 0, 1)
	case "prophecy.wormhole":
		ae.config.WormholeChance = clamp(value, 0, 1)
	case "meta.adaptation_rate":
		ae.config.AdaptationRate = clamp(value, 0, 1)
	case "meta.stability":
		ae.config.StabilityBias = clamp(value, 0, 1)
	default:
		return fmt.Errorf("unknown parameter: %s", name)
	}

	return nil
}

// GetParam gets a parameter by name (like sysctl)
func (ae *AdaptiveEngine) GetParam(name string) (float32, error) {
	ae.config.mu.RLock()
	defer ae.config.mu.RUnlock()

	switch name {
	case "trauma.weight":
		return ae.config.TraumaWeight, nil
	case "trauma.decay":
		return ae.config.TraumaDecayRate, nil
	case "trauma.threshold":
		return ae.config.TraumaThreshold, nil
	case "drift.speed":
		return ae.config.DriftSpeed, nil
	case "drift.inertia":
		return ae.config.DriftInertia, nil
	case "attention.focus_decay":
		return ae.config.FocusDecay, nil
	case "attention.wander_threshold":
		return ae.config.WanderThreshold, nil
	case "prophecy.destiny":
		return ae.config.DestinyStrength, nil
	case "prophecy.wormhole":
		return ae.config.WormholeChance, nil
	case "meta.adaptation_rate":
		return ae.config.AdaptationRate, nil
	case "meta.stability":
		return ae.config.StabilityBias, nil
	default:
		return 0, fmt.Errorf("unknown parameter: %s", name)
	}
}

// ListParams returns all parameter names
func (ae *AdaptiveEngine) ListParams() []string {
	return []string{
		"trauma.weight",
		"trauma.decay",
		"trauma.threshold",
		"drift.speed",
		"drift.inertia",
		"attention.focus_decay",
		"attention.wander_threshold",
		"prophecy.destiny",
		"prophecy.wormhole",
		"meta.adaptation_rate",
		"meta.stability",
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL ADAPTIVE ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalAdaptive *AdaptiveEngine
	adaptiveMu     sync.Mutex
)

// GetAdaptiveEngine returns the global adaptive engine
func GetAdaptiveEngine() *AdaptiveEngine {
	adaptiveMu.Lock()
	defer adaptiveMu.Unlock()

	if globalAdaptive == nil {
		globalAdaptive = NewAdaptiveEngine()
	}
	return globalAdaptive
}

// AdaptGlobal runs adaptation on global state
func AdaptGlobal() {
	engine := GetAdaptiveEngine()
	state := Global().State
	engine.Adapt(state)
}
