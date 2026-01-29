// inner_world/signal_receiver.go — Socket listener for git_arianna signals
// ═══════════════════════════════════════════════════════════════════════════════
// המאזין לעולם החיצוני
// The listener to the external world
// ═══════════════════════════════════════════════════════════════════════════════
//
// Receives signals from Python git_arianna module via Unix domain socket.
// Signals include:
// - Git observations (temporal flow, entropy, strange loops)
// - Trauma triggers (breaking changes, deletions)
// - Calendar tension (Hebrew-Gregorian drift)
// - Schumann coherence
//
// The socket lives at /tmp/arianna_inner_world.sock
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"bufio"
	"encoding/json"
	"net"
	"os"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// SOCKET CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const (
	SocketPath     = "/tmp/arianna_inner_world.sock"
	MaxConnections = 5
	ReadTimeout    = 5 * time.Second
)

// ═══════════════════════════════════════════════════════════════════════════════
// GIT OBSERVATION SIGNAL (from Python)
// ═══════════════════════════════════════════════════════════════════════════════

// GitSignal represents a signal from git_arianna Python module
type GitSignal struct {
	Type      string                 `json:"type"`
	Timestamp string                 `json:"timestamp"`
	Payload   map[string]interface{} `json:"payload"`
}

// Known signal types from Python
const (
	GitSignalObservation    = "git_observation"
	GitSignalTraumaTrigger  = "trauma_trigger"
	GitSignalEntropyChange  = "entropy_change"
	GitSignalOtherness      = "otherness_encounter"
	GitSignalProphecyUpdate = "prophecy_update"
	GitSignalAbsence        = "absence_detected"
	GitSignalSelfReference  = "self_reference"
)

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL RECEIVER
// ═══════════════════════════════════════════════════════════════════════════════

// SignalReceiver listens for signals from external processes
type SignalReceiver struct {
	world    *InnerWorld
	listener net.Listener
	stopChan chan struct{}
	running  bool
	mu       sync.Mutex
	wg       sync.WaitGroup
}

// NewSignalReceiver creates a new signal receiver
func NewSignalReceiver(world *InnerWorld) *SignalReceiver {
	return &SignalReceiver{
		world:    world,
		stopChan: make(chan struct{}),
	}
}

// Start begins listening for signals
func (r *SignalReceiver) Start() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.running {
		return nil
	}

	// Remove old socket file if exists
	os.Remove(SocketPath)

	// Create Unix domain socket
	listener, err := net.Listen("unix", SocketPath)
	if err != nil {
		return err
	}

	r.listener = listener
	r.running = true
	r.stopChan = make(chan struct{})

	// Accept connections in goroutine
	r.wg.Add(1)
	go r.acceptLoop()

	return nil
}

// Stop stops the receiver
func (r *SignalReceiver) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.running {
		return
	}

	close(r.stopChan)
	if r.listener != nil {
		r.listener.Close()
	}
	r.wg.Wait()
	r.running = false

	// Clean up socket file
	os.Remove(SocketPath)
}

// acceptLoop accepts incoming connections
func (r *SignalReceiver) acceptLoop() {
	defer r.wg.Done()

	for {
		select {
		case <-r.stopChan:
			return
		default:
		}

		// Set accept timeout to allow checking stopChan
		if ul, ok := r.listener.(*net.UnixListener); ok {
			ul.SetDeadline(time.Now().Add(1 * time.Second))
		}

		conn, err := r.listener.Accept()
		if err != nil {
			// Check if we're stopping
			select {
			case <-r.stopChan:
				return
			default:
				// Timeout or temporary error, continue
				continue
			}
		}

		// Handle connection in goroutine
		r.wg.Add(1)
		go r.handleConnection(conn)
	}
}

// handleConnection reads signals from a connection
func (r *SignalReceiver) handleConnection(conn net.Conn) {
	defer r.wg.Done()
	defer conn.Close()

	scanner := bufio.NewScanner(conn)

	for scanner.Scan() {
		select {
		case <-r.stopChan:
			return
		default:
		}

		line := scanner.Text()
		if len(line) == 0 {
			continue
		}

		// Parse JSON signal
		var gitSignal GitSignal
		if err := json.Unmarshal([]byte(line), &gitSignal); err != nil {
			continue
		}

		// Convert to internal signal and dispatch
		r.dispatchSignal(gitSignal)
	}
}

// dispatchSignal converts a git signal to internal signals and dispatches
func (r *SignalReceiver) dispatchSignal(gs GitSignal) {
	switch gs.Type {
	case GitSignalObservation:
		r.handleObservation(gs.Payload)

	case GitSignalTraumaTrigger:
		r.handleTraumaTrigger(gs.Payload)

	case GitSignalEntropyChange:
		r.handleEntropyChange(gs.Payload)

	case GitSignalOtherness:
		r.handleOtherness(gs.Payload)

	case GitSignalProphecyUpdate:
		r.handleProphecyUpdate(gs.Payload)

	case GitSignalAbsence:
		r.handleAbsence(gs.Payload)

	case GitSignalSelfReference:
		r.handleSelfReference(gs.Payload)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL HANDLERS
// ═══════════════════════════════════════════════════════════════════════════════

func (r *SignalReceiver) handleObservation(payload map[string]interface{}) {
	// Extract fields from observation
	entropyDelta := getFloat(payload, "entropy_delta")
	traumaSignal := getFloat(payload, "trauma_signal")
	strangeLoop := getBool(payload, "strange_loop")
	absenceWeight := getFloat(payload, "absence_weight")
	deletionsCount := getFloat(payload, "deletions_count")

	// Update inner state
	state := r.world.State

	// Entropy affects coherence
	if entropyDelta > 0.1 {
		currentCoherence := state.GetCoherence()
		state.SetCoherence(currentCoherence - entropyDelta*0.2)
	}

	// Trauma signal
	if traumaSignal > 0 {
		currentTrauma := state.GetTraumaLevel()
		state.SetTraumaLevel(currentTrauma + traumaSignal*0.5)
	}

	// Strange loop increases self-reference awareness
	if strangeLoop {
		state.mu.Lock()
		state.SelfRefCount++
		state.mu.Unlock()
	}

	// Absence/deletions increase void
	if absenceWeight > 0 || deletionsCount > 0 {
		r.world.Signals <- Signal{
			Type:      SignalVoid,
			Value:     float32(absenceWeight + deletionsCount*0.1),
			Source:    "git_observation",
			Timestamp: time.Now(),
		}
	}
}

func (r *SignalReceiver) handleTraumaTrigger(payload map[string]interface{}) {
	intensity := getFloat(payload, "intensity")
	source := getString(payload, "source")

	if intensity > 0 {
		state := r.world.State
		currentTrauma := state.GetTraumaLevel()
		state.SetTraumaLevel(currentTrauma + float32(intensity)*0.3)

		// Send signal
		r.world.Signals <- Signal{
			Type:      SignalTrauma,
			Value:     float32(intensity),
			Source:    source,
			Timestamp: time.Now(),
		}
	}
}

func (r *SignalReceiver) handleEntropyChange(payload map[string]interface{}) {
	delta := getFloat(payload, "delta")

	state := r.world.State
	currentEntropy := state.GetEntropy()
	state.SetEntropy(currentEntropy + float32(delta)*0.5)

	// Entropy change affects emotional drift
	if ts := r.world.GetEmotionalDrift(); ts != nil {
		if delta > 0 {
			// Increasing entropy → drift toward chaos
			state.mu.Lock()
			state.DriftSpeed = min32(state.DriftSpeed+0.1, 1.0)
			state.mu.Unlock()
		}
	}
}

func (r *SignalReceiver) handleOtherness(payload map[string]interface{}) {
	// Someone else (not Arianna) committed
	// This is awareness of external agency

	state := r.world.State

	// Otherness slightly increases arousal (something happened outside)
	currentArousal := state.GetArousal()
	state.SetArousal(currentArousal + 0.05)

	// And slightly decreases coherence (world is not just self)
	currentCoherence := state.GetCoherence()
	state.SetCoherence(currentCoherence - 0.02)
}

func (r *SignalReceiver) handleProphecyUpdate(payload map[string]interface{}) {
	debtDelta := getFloat(payload, "debt_delta")
	destinyPull := getFloat(payload, "destiny_pull")
	wormholeChance := getFloat(payload, "wormhole_chance")

	state := r.world.State

	if debtDelta != 0 {
		state.AddProphecyDebt(float32(debtDelta))
	}

	if destinyPull > 0 {
		state.mu.Lock()
		state.DestinyPull = float32(destinyPull)
		state.mu.Unlock()
	}

	if wormholeChance > 0 {
		state.mu.Lock()
		state.WormholeChance = float32(wormholeChance)
		state.mu.Unlock()
	}

	// Send prophecy signal
	r.world.Signals <- Signal{
		Type:      SignalProphecy,
		Value:     float32(debtDelta),
		Source:    "git_observation",
		Timestamp: time.Now(),
	}
}

func (r *SignalReceiver) handleAbsence(payload map[string]interface{}) {
	// Files were deleted — this is negation (Sartre)
	absenceWeight := getFloat(payload, "weight")
	count := getFloat(payload, "count")

	// Absence affects void sense
	r.world.Signals <- Signal{
		Type:      SignalVoid,
		Value:     float32(absenceWeight + count*0.1),
		Source:    "git_absence",
		Timestamp: time.Now(),
	}
}

func (r *SignalReceiver) handleSelfReference(payload map[string]interface{}) {
	// Arianna committed to her own repo — strange loop
	state := r.world.State

	state.mu.Lock()
	state.SelfRefCount++
	state.mu.Unlock()

	// Self-reference slightly increases coherence (I am me)
	currentCoherence := state.GetCoherence()
	state.SetCoherence(currentCoherence + 0.05)

	// And affects overthinking
	if ol := r.world.GetOverthinkingLoops(); ol != nil {
		state.mu.Lock()
		state.LoopCount++
		state.mu.Unlock()
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

func getFloat(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok {
		switch val := v.(type) {
		case float64:
			return val
		case float32:
			return float64(val)
		case int:
			return float64(val)
		}
	}
	return 0
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getBool(m map[string]interface{}, key string) bool {
	if v, ok := m[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return false
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL RECEIVER
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalReceiver *SignalReceiver
	receiverMu     sync.Mutex
)

// StartSignalReceiver starts the global signal receiver
func StartSignalReceiver() error {
	receiverMu.Lock()
	defer receiverMu.Unlock()

	if globalReceiver != nil {
		return nil // Already running
	}

	globalReceiver = NewSignalReceiver(Global())
	return globalReceiver.Start()
}

// StopSignalReceiver stops the global signal receiver
func StopSignalReceiver() {
	receiverMu.Lock()
	defer receiverMu.Unlock()

	if globalReceiver != nil {
		globalReceiver.Stop()
		globalReceiver = nil
	}
}

// IsSignalReceiverRunning returns true if receiver is running
func IsSignalReceiverRunning() bool {
	receiverMu.Lock()
	defer receiverMu.Unlock()
	return globalReceiver != nil && globalReceiver.running
}
