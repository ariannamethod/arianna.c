// inner_world/signal_receiver_test.go — Tests for git_arianna signal receiver
// ═══════════════════════════════════════════════════════════════════════════════

package main

import (
	"encoding/json"
	"net"
	"testing"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

func sendSignal(t *testing.T, signal GitSignal) {
	conn, err := net.Dial("unix", SocketPath)
	if err != nil {
		t.Fatalf("Failed to connect to socket: %v", err)
	}
	defer conn.Close()

	data, err := json.Marshal(signal)
	if err != nil {
		t.Fatalf("Failed to marshal signal: %v", err)
	}

	_, err = conn.Write(append(data, '\n'))
	if err != nil {
		t.Fatalf("Failed to write signal: %v", err)
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

func TestSignalReceiverStartStop(t *testing.T) {
	world := NewInnerWorld()
	receiver := NewSignalReceiver(world)

	// Start
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}

	if !receiver.running {
		t.Error("Receiver should be running after Start")
	}

	// Stop
	receiver.Stop()

	if receiver.running {
		t.Error("Receiver should not be running after Stop")
	}
}

func TestSignalReceiverConnection(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	// Try to connect
	conn, err := net.Dial("unix", SocketPath)
	if err != nil {
		t.Fatalf("Failed to connect to socket: %v", err)
	}
	conn.Close()
}

func TestHandleObservation(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	// Initial state
	initialCoherence := world.State.GetCoherence()
	initialTrauma := world.State.GetTraumaLevel()

	// Send observation with high entropy and trauma
	signal := GitSignal{
		Type:      GitSignalObservation,
		Timestamp: time.Now().Format(time.RFC3339),
		Payload: map[string]interface{}{
			"entropy_delta":  0.5,
			"trauma_signal":  0.3,
			"strange_loop":   true,
			"absence_weight": 0.2,
		},
	}

	sendSignal(t, signal)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Check state changed
	newCoherence := world.State.GetCoherence()
	newTrauma := world.State.GetTraumaLevel()

	if newCoherence >= initialCoherence {
		t.Logf("Coherence: %f -> %f (expected decrease)", initialCoherence, newCoherence)
	}

	if newTrauma <= initialTrauma {
		t.Logf("Trauma: %f -> %f (expected increase)", initialTrauma, newTrauma)
	}
}

func TestHandleTraumaTrigger(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	initialTrauma := world.State.GetTraumaLevel()

	signal := GitSignal{
		Type:      GitSignalTraumaTrigger,
		Timestamp: time.Now().Format(time.RFC3339),
		Payload: map[string]interface{}{
			"intensity": 0.8,
			"source":    "breaking_change",
		},
	}

	sendSignal(t, signal)
	time.Sleep(100 * time.Millisecond)

	newTrauma := world.State.GetTraumaLevel()
	if newTrauma <= initialTrauma {
		t.Errorf("Trauma should increase after trauma trigger: %f -> %f", initialTrauma, newTrauma)
	}
}

func TestHandleEntropyChange(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	initialEntropy := world.State.GetEntropy()

	signal := GitSignal{
		Type:      GitSignalEntropyChange,
		Timestamp: time.Now().Format(time.RFC3339),
		Payload: map[string]interface{}{
			"delta": 0.3,
		},
	}

	sendSignal(t, signal)
	time.Sleep(100 * time.Millisecond)

	newEntropy := world.State.GetEntropy()
	if newEntropy <= initialEntropy {
		t.Errorf("Entropy should increase: %f -> %f", initialEntropy, newEntropy)
	}
}

func TestHandleProphecyUpdate(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	initialDebt := world.State.GetProphecyDebt()

	signal := GitSignal{
		Type:      GitSignalProphecyUpdate,
		Timestamp: time.Now().Format(time.RFC3339),
		Payload: map[string]interface{}{
			"debt_delta":     0.5,
			"destiny_pull":   0.7,
			"wormhole_chance": 0.1,
		},
	}

	sendSignal(t, signal)
	time.Sleep(100 * time.Millisecond)

	newDebt := world.State.GetProphecyDebt()
	if newDebt <= initialDebt {
		t.Errorf("Prophecy debt should increase: %f -> %f", initialDebt, newDebt)
	}

	world.State.mu.RLock()
	destinyPull := world.State.DestinyPull
	wormholeChance := world.State.WormholeChance
	world.State.mu.RUnlock()

	if destinyPull != 0.7 {
		t.Errorf("Destiny pull should be 0.7, got %f", destinyPull)
	}

	if wormholeChance != 0.1 {
		t.Errorf("Wormhole chance should be 0.1, got %f", wormholeChance)
	}
}

func TestHandleSelfReference(t *testing.T) {
	world := NewInnerWorld()
	world.Start()
	defer world.Stop()

	receiver := NewSignalReceiver(world)
	err := receiver.Start()
	if err != nil {
		t.Fatalf("Failed to start receiver: %v", err)
	}
	defer receiver.Stop()

	world.State.mu.RLock()
	initialSelfRef := world.State.SelfRefCount
	world.State.mu.RUnlock()

	signal := GitSignal{
		Type:      GitSignalSelfReference,
		Timestamp: time.Now().Format(time.RFC3339),
		Payload:   map[string]interface{}{},
	}

	sendSignal(t, signal)
	time.Sleep(100 * time.Millisecond)

	world.State.mu.RLock()
	newSelfRef := world.State.SelfRefCount
	world.State.mu.RUnlock()

	if newSelfRef <= initialSelfRef {
		t.Errorf("Self-reference count should increase: %d -> %d", initialSelfRef, newSelfRef)
	}
}

func TestHelperFunctions(t *testing.T) {
	payload := map[string]interface{}{
		"float_val":  3.14,
		"int_val":    42,
		"string_val": "hello",
		"bool_val":   true,
	}

	// Test getFloat
	if f := getFloat(payload, "float_val"); f != 3.14 {
		t.Errorf("getFloat: expected 3.14, got %f", f)
	}

	if f := getFloat(payload, "int_val"); f != 42.0 {
		t.Errorf("getFloat from int: expected 42.0, got %f", f)
	}

	if f := getFloat(payload, "missing"); f != 0 {
		t.Errorf("getFloat missing: expected 0, got %f", f)
	}

	// Test getString
	if s := getString(payload, "string_val"); s != "hello" {
		t.Errorf("getString: expected 'hello', got '%s'", s)
	}

	if s := getString(payload, "missing"); s != "" {
		t.Errorf("getString missing: expected '', got '%s'", s)
	}

	// Test getBool
	if b := getBool(payload, "bool_val"); !b {
		t.Error("getBool: expected true")
	}

	if b := getBool(payload, "missing"); b {
		t.Error("getBool missing: expected false")
	}
}

func TestGlobalReceiverStartStop(t *testing.T) {
	// Initialize world first
	Init()
	defer Shutdown()

	// Start receiver
	err := StartSignalReceiver()
	if err != nil {
		t.Fatalf("Failed to start global receiver: %v", err)
	}

	if !IsSignalReceiverRunning() {
		t.Error("Global receiver should be running")
	}

	// Stop
	StopSignalReceiver()

	if IsSignalReceiverRunning() {
		t.Error("Global receiver should not be running after stop")
	}
}
