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

// Start begins the inner world. async=true: each process self-ticks in its own
// goroutine (for a standalone / C-host run). async=false: the processes do NOT
// self-tick — the caller drives everything through iw.Step on a single clock, so
// Step/ProcessText (both under iw.mu) are the only writers and there is no race.
func (iw *InnerWorld) Start(async bool) {
	iw.mu.Lock()
	defer iw.mu.Unlock()

	if iw.running {
		return
	}
	iw.async = async

	// Recreate channels on restart (fixes channel reuse after Stop)
	iw.stopChan = make(chan struct{})
	iw.Signals = make(chan Signal, 100)
	iw.Commands = make(chan Command, 10)

	// Clear old processes
	iw.processes = iw.processes[:0]

	// Create and start all processes
	for _, constructor := range registeredProcesses {
		proc := constructor()
		iw.processes = append(iw.processes, proc)
		// Start synchronously to set world field immediately
		proc.Start(iw)
	}

	// The processes read iw.Signals directly in their own Step()/run() select, so a
	// separate routeSignals drainer would COMPETE with them and discard signals the
	// intended process should have seen — it is not started. (Was: go routeSignals.)

	// Start command handler — joined on Stop (added to the waitgroup).
	iw.wg.Add(1)
	go iw.handleCommands()

	iw.running = true
}

// Stop gracefully stops all processes
func (iw *InnerWorld) Stop() {
	iw.mu.Lock()
	if !iw.running {
		iw.mu.Unlock()
		return
	}
	iw.running = false
	close(iw.stopChan) // signal stop
	for _, proc := range iw.processes {
		proc.Stop()
	}
	iw.mu.Unlock()
	// Join handleCommands OUTSIDE iw.mu — it may be mid-command (CmdReset/CmdStep)
	// that needs iw.mu, so waiting while holding the lock would deadlock.
	iw.wg.Wait()
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
	defer iw.wg.Done()
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

// GetProcess returns a process by name. No lock: iw.processes is immutable during
// a run (appended in Start, cleared in Stop, both under iw.mu) and is only read
// here and in Step — concurrent reads don't race. Taking iw.mu here would re-enter
// it when called from ProcessText (which already holds iw.mu) and deadlock.
func (iw *InnerWorld) GetProcess(name string) Process {
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

// ResyncMood re-syncs the mood processes' private state from the (just-loaded)
// State, so a LoadState that runs AFTER Start() is not clobbered by the defaults the
// processes snapshot at Start. Lock-free (GetProcess takes no lock), so it is safe
// to call under iw.mu.
func (iw *InnerWorld) ResyncMood() {
	if ed := iw.GetEmotionalDrift(); ed != nil {
		ed.Resync()
	}
	if pd := iw.GetProphecyDebt(); pd != nil {
		pd.Resync()
	}
}

// RestoreMood loads the persisted state and re-syncs the processes ATOMICALLY with
// respect to the ticker — both run under iw.mu, which Step also takes, so no Step can
// fire between the load and the resync and write a stale (default) private value back
// into State. Returns the last dream. Use this instead of LoadState+ResyncMood.
func (iw *InnerWorld) RestoreMood(path string) string {
	iw.mu.Lock()
	defer iw.mu.Unlock()
	ld := iw.LoadState(path)
	iw.ResyncMood()
	return ld
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
	iw.mu.Lock()
	defer iw.mu.Unlock()
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
	Global().Start(true) // C-host path: processes self-tick
}

// Shutdown stops the global inner world
func Shutdown() {
	// take the world out under globalMu, then Stop it OUTSIDE the lock: Stop takes
	// iw.mu, and Step (holding iw.mu) calls AdaptGlobal → Global() → globalMu, so
	// holding globalMu across Stop would invert the lock order and can deadlock.
	globalMu.Lock()
	w := globalWorld
	globalWorld = nil
	globalMu.Unlock()
	if w != nil {
		w.Stop()
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
