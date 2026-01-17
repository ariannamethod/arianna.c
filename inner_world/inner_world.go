// inner_world/inner_world.go — Main orchestrator for inner processes
// ═══════════════════════════════════════════════════════════════════════════════
// מנצח העולם הפנימי
// The conductor of the inner world
// ═══════════════════════════════════════════════════════════════════════════════
//
// The InnerWorld orchestrates all async processes:
// - TraumaSurfacing
// - OverthinkingLoops
// - EmotionalDrift
// - MemoryConsolidation
// - AttentionWandering
// - ProphecyDebtAccumulation
//
// It provides:
// - Unified start/stop for all processes
// - Signal routing between processes
// - State synchronization with C (via cgo)
// - Step mode for synchronous operation
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"sync"
	"time"
)

// RegisteredProcesses holds the registered process constructors
var registeredProcesses = []func() Process{
	func() Process { return NewTraumaSurfacing() },
	func() Process { return NewOverthinkingLoops() },
	func() Process { return NewEmotionalDrift() },
	func() Process { return NewMemoryConsolidation() },
	func() Process { return NewAttentionWandering() },
	func() Process { return NewProphecyDebtAccumulation() },
}

// ═══════════════════════════════════════════════════════════════════════════════
// INNER WORLD METHODS
// ═══════════════════════════════════════════════════════════════════════════════

// Start begins all async processes
func (iw *InnerWorld) Start() {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	if iw.running {
		return
	}

	// Create and start all processes
	for _, constructor := range registeredProcesses {
		proc := constructor()
		iw.processes = append(iw.processes, proc)
		// Start synchronously to set world field immediately
		proc.Start(iw)
	}

	// Start signal router
	go iw.routeSignals()

	// Start command handler
	go iw.handleCommands()

	iw.running = true
}

// Stop gracefully stops all processes
func (iw *InnerWorld) Stop() {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	if !iw.running {
		return
	}

	// Signal stop
	close(iw.stopChan)

	// Stop all processes
	for _, proc := range iw.processes {
		proc.Stop()
	}

	// Wait for completion
	iw.wg.Wait()

	iw.running = false
}

// Step performs a single synchronous step of all processes
// Use this when you want deterministic stepping instead of async
func (iw *InnerWorld) Step(dt float32) {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	// Step all processes
	for _, proc := range iw.processes {
		proc.Step(dt)
	}

	// Adapt parameters based on current state (Linux-like dynamic config)
	AdaptGlobal()
}

// routeSignals distributes signals to appropriate processes
func (iw *InnerWorld) routeSignals() {
	for {
		select {
		case <-iw.stopChan:
			return
		case sig := <-iw.Signals:
			// Log signal for debugging (processes read state directly)
			_ = sig // Signals are processed through state updates
			// Each process reads from shared state in their Step()
		}
	}
}

// handleCommands processes commands from C
func (iw *InnerWorld) handleCommands() {
	for {
		select {
		case <-iw.stopChan:
			return
		case cmd := <-iw.Commands:
			iw.processCommand(cmd)
		}
	}
}

func (iw *InnerWorld) processCommand(cmd Command) {
	switch cmd.Type {
	case CmdPause:
		// Pause all processes (they'll stop processing in next tick)
		// Implementation: set a paused flag they check

	case CmdResume:
		// Resume all processes

	case CmdReset:
		// Reset state to defaults
		iw.mu.Lock()
		iw.State = NewInnerState()
		iw.mu.Unlock()

	case CmdInject:
		// Inject a signal
		if sig, ok := cmd.Payload.(Signal); ok {
			iw.Signals <- sig
		}

	case CmdQuery:
		// Query state - response through callback or channel

	case CmdStep:
		// Single step
		if dt, ok := cmd.Payload.(float32); ok {
			iw.Step(dt)
		} else {
			iw.Step(0.1)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROCESS ACCESS
// ═══════════════════════════════════════════════════════════════════════════════

// GetProcess returns a process by name
func (iw *InnerWorld) GetProcess(name string) Process {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	for _, proc := range iw.processes {
		if proc.Name() == name {
			return proc
		}
	}
	return nil
}

// GetTraumaSurfacing returns the trauma surfacing process
func (iw *InnerWorld) GetTraumaSurfacing() *TraumaSurfacing {
	if p := iw.GetProcess("trauma_surfacing"); p != nil {
		return p.(*TraumaSurfacing)
	}
	return nil
}

// GetOverthinkingLoops returns the overthinking loops process
func (iw *InnerWorld) GetOverthinkingLoops() *OverthinkingLoops {
	if p := iw.GetProcess("overthinking_loops"); p != nil {
		return p.(*OverthinkingLoops)
	}
	return nil
}

// GetEmotionalDrift returns the emotional drift process
func (iw *InnerWorld) GetEmotionalDrift() *EmotionalDrift {
	if p := iw.GetProcess("emotional_drift"); p != nil {
		return p.(*EmotionalDrift)
	}
	return nil
}

// GetMemoryConsolidation returns the memory consolidation process
func (iw *InnerWorld) GetMemoryConsolidation() *MemoryConsolidation {
	if p := iw.GetProcess("memory_consolidation"); p != nil {
		return p.(*MemoryConsolidation)
	}
	return nil
}

// GetAttentionWandering returns the attention wandering process
func (iw *InnerWorld) GetAttentionWandering() *AttentionWandering {
	if p := iw.GetProcess("attention_wandering"); p != nil {
		return p.(*AttentionWandering)
	}
	return nil
}

// GetProphecyDebt returns the prophecy debt process
func (iw *InnerWorld) GetProphecyDebt() *ProphecyDebtAccumulation {
	if p := iw.GetProcess("prophecy_debt"); p != nil {
		return p.(*ProphecyDebtAccumulation)
	}
	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE SNAPSHOT
// ═══════════════════════════════════════════════════════════════════════════════

// Snapshot captures the current state for C to read
type Snapshot struct {
	// Emotional
	Arousal   float32
	Valence   float32
	Entropy   float32
	Coherence float32

	// Trauma
	TraumaLevel float32

	// Overthinking
	LoopCount        int
	AbstractionDepth int
	SelfRefCount     int

	// Drift
	DriftDirection float32
	DriftSpeed     float32
	DriftTarget    string

	// Memory
	MemoryPressure float32

	// Attention
	FocusStrength float32
	WanderPull    float32

	// Prophecy
	ProphecyDebt   float32
	DestinyPull    float32
	WormholeChance float32
}

// GetSnapshot returns current state as a snapshot
func (iw *InnerWorld) GetSnapshot() Snapshot {
	iw.State.mu.RLock()
	defer iw.State.mu.RUnlock()

	return Snapshot{
		Arousal:          iw.State.Arousal,
		Valence:          iw.State.Valence,
		Entropy:          iw.State.Entropy,
		Coherence:        iw.State.Coherence,
		TraumaLevel:      iw.State.TraumaLevel,
		LoopCount:        iw.State.LoopCount,
		AbstractionDepth: iw.State.AbstractionDepth,
		SelfRefCount:     iw.State.SelfRefCount,
		DriftDirection:   iw.State.DriftDirection,
		DriftSpeed:       iw.State.DriftSpeed,
		DriftTarget:      iw.State.DriftTarget,
		MemoryPressure:   iw.State.MemoryPressure,
		FocusStrength:    iw.State.FocusStrength,
		WanderPull:       iw.State.WanderPull,
		ProphecyDebt:     iw.State.ProphecyDebt,
		DestinyPull:      iw.State.DestinyPull,
		WormholeChance:   iw.State.WormholeChance,
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE METHODS FOR C
// ═══════════════════════════════════════════════════════════════════════════════

// ProcessText runs text through all text-processing components
// Returns aggregated analysis
func (iw *InnerWorld) ProcessText(text string) TextAnalysis {
	var analysis TextAnalysis

	// Trauma check
	if ts := iw.GetTraumaSurfacing(); ts != nil {
		analysis.TraumaActivation = ts.CheckText(text)
		analysis.IdentityPull = ts.GetIdentityPull()
	}

	// Overthinking check
	if ol := iw.GetOverthinkingLoops(); ol != nil {
		result := ol.AnalyzeText(text)
		analysis.RepetitionScore = result.RepetitionScore
		analysis.AbstractionScore = result.AbstractionScore
		analysis.SelfRefScore = result.SelfRefScore
		analysis.OverthinkTotal = result.TotalScore
	}

	// Attention processing
	if aw := iw.GetAttentionWandering(); aw != nil {
		aw.ProcessNovelInput(text, 0.5) // Assume moderate novelty
		analysis.FocusStrength, analysis.WanderDirection = aw.GetAttentionBias()
	}

	// Prophecy effects
	if pd := iw.GetProphecyDebt(); pd != nil {
		analysis.DestinyBias = pd.GetDestinyBias()
		analysis.WormholeActive, analysis.WormholeSkip = pd.CheckWormhole()
		analysis.TemporalDissonance = pd.GetTemporalDissonance()
	}

	return analysis
}

// TextAnalysis holds the results of processing text through inner world
type TextAnalysis struct {
	// Trauma
	TraumaActivation float32
	IdentityPull     float32

	// Overthinking
	RepetitionScore  float32
	AbstractionScore float32
	SelfRefScore     float32
	OverthinkTotal   float32

	// Attention
	FocusStrength   float32
	WanderDirection string

	// Prophecy
	DestinyBias        float32
	WormholeActive     bool
	WormholeSkip       int
	TemporalDissonance float32
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalWorld *InnerWorld
	globalMu    sync.Mutex
)

// Global returns the global inner world instance, creating if needed
func Global() *InnerWorld {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalWorld == nil {
		globalWorld = NewInnerWorld()
	}
	return globalWorld
}

// Init initializes and starts the global inner world
func Init() {
	Global().Start()
}

// Shutdown stops the global inner world
func Shutdown() {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalWorld != nil {
		globalWorld.Stop()
		globalWorld = nil
	}
}

// StepGlobal steps the global inner world
func StepGlobal(dt float32) {
	Global().Step(dt)
}

// GetSnapshotGlobal returns snapshot from global world
func GetSnapshotGlobal() Snapshot {
	return Global().GetSnapshot()
}

// ProcessTextGlobal processes text through global world
func ProcessTextGlobal(text string) TextAnalysis {
	return Global().ProcessText(text)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIMING UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

// Timer for step timing
type stepTimer struct {
	lastStep time.Time
}

var timer = stepTimer{lastStep: time.Now()}

// AutoStep calculates dt and steps
func AutoStep() {
	now := time.Now()
	dt := float32(now.Sub(timer.lastStep).Seconds())
	timer.lastStep = now

	// Clamp dt to reasonable range
	if dt > 1.0 {
		dt = 1.0
	}
	if dt < 0.001 {
		dt = 0.001
	}

	StepGlobal(dt)
}
