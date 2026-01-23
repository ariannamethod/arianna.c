// vagus_test.zig — Comprehensive tests for the Wandering Nerve
// ═══════════════════════════════════════════════════════════════════════════════
// Target: 100% coverage of vagus.zig
// ═══════════════════════════════════════════════════════════════════════════════

const std = @import("std");
const vagus = @import("vagus.zig");

const Signal = vagus.Signal;
const Source = vagus.Source;
const SignalType = vagus.SignalType;
const SharedState = vagus.SharedState;
const VagusNerve = vagus.VagusNerve;
const RingBuffer = vagus.RingBuffer;
const CrossFireMatrix = vagus.CrossFireMatrix;

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "Signal size is exactly 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Signal));
}

test "Signal.now creates valid signal" {
    const sig = Signal.now(.cloud, .arousal, 0.75);

    try std.testing.expectEqual(Source.cloud, sig.source);
    try std.testing.expectEqual(SignalType.arousal, sig.signal_type);
    try std.testing.expectEqual(@as(f32, 0.75), sig.value);
    try std.testing.expect(sig.timestamp_us > 0);
}

test "Signal covers all sources" {
    const sources = [_]Source{
        .arianna, .cloud, .inner_world, .sartre,
        .delta, .pandora, .limpha, .external,
    };

    for (sources, 0..) |src, i| {
        try std.testing.expectEqual(@as(u8, @intCast(i)), @intFromEnum(src));
    }
}

test "Signal covers emotional types" {
    const emotional_types = [_]SignalType{
        .arousal, .valence, .warmth, .void_level, .tension, .sacred,
    };

    for (emotional_types) |t| {
        const sig = Signal.now(.cloud, t, 0.5);
        try std.testing.expectEqual(t, sig.signal_type);
    }
}

test "Signal covers cognitive types" {
    const cognitive_types = [_]SignalType{
        .coherence, .entropy, .focus, .abstraction,
    };

    for (cognitive_types) |t| {
        const sig = Signal.now(.arianna, t, 0.5);
        try std.testing.expectEqual(t, sig.signal_type);
    }
}

test "Signal covers temporal types" {
    const temporal_types = [_]SignalType{
        .drift_direction, .drift_speed, .prophecy_debt, .destiny_pull, .wormhole,
    };

    for (temporal_types) |t| {
        const sig = Signal.now(.inner_world, t, 0.5);
        try std.testing.expectEqual(t, sig.signal_type);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED STATE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "SharedState fits in a page" {
    try std.testing.expect(@sizeOf(SharedState) <= 4096);
}

test "SharedState default values" {
    var state = SharedState{};

    try std.testing.expectEqual(@as(f32, 0.3), state.arousal);
    try std.testing.expectEqual(@as(f32, 0.0), state.valence);
    try std.testing.expectEqual(@as(f32, 0.8), state.coherence);
    try std.testing.expectEqual(@as(f32, 0.2), state.entropy);
    try std.testing.expectEqual(@as(f32, 0.0), state.trauma_level);
    try std.testing.expectEqual(@as(f32, 0.0), state.prophecy_debt);
    try std.testing.expectEqual(@as(u32, 1), state.vagus_version);
}

test "SharedState atomic arousal get/set" {
    var state = SharedState{};

    state.setArousal(0.7);
    try std.testing.expectEqual(@as(f32, 0.7), state.getArousal());

    // Test clamping
    state.setArousal(1.5);
    try std.testing.expectEqual(@as(f32, 1.0), state.getArousal());

    state.setArousal(-0.5);
    try std.testing.expectEqual(@as(f32, 0.0), state.getArousal());
}

test "SharedState getChambers returns all 6" {
    var state = SharedState{};
    state.chamber_warmth = 0.1;
    state.chamber_void = 0.2;
    state.chamber_tension = 0.3;
    state.chamber_sacred = 0.4;
    state.chamber_flow = 0.5;
    state.chamber_complex = 0.6;

    const chambers = state.getChambers();

    try std.testing.expectEqual(@as(f32, 0.1), chambers[0]);
    try std.testing.expectEqual(@as(f32, 0.2), chambers[1]);
    try std.testing.expectEqual(@as(f32, 0.3), chambers[2]);
    try std.testing.expectEqual(@as(f32, 0.4), chambers[3]);
    try std.testing.expectEqual(@as(f32, 0.5), chambers[4]);
    try std.testing.expectEqual(@as(f32, 0.6), chambers[5]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RING BUFFER TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "RingBuffer push and pop single item" {
    var ring = RingBuffer(Signal, 16){};

    const sig = Signal.now(.cloud, .arousal, 0.5);
    try std.testing.expect(ring.push(sig));

    const popped = ring.pop();
    try std.testing.expect(popped != null);
    try std.testing.expectEqual(@as(f32, 0.5), popped.?.value);
}

test "RingBuffer pop empty returns null" {
    var ring = RingBuffer(Signal, 16){};

    const popped = ring.pop();
    try std.testing.expect(popped == null);
}

test "RingBuffer peek without consuming" {
    var ring = RingBuffer(Signal, 16){};

    const sig = Signal.now(.cloud, .valence, 0.3);
    _ = ring.push(sig);

    // Peek should return item
    const peeked = ring.peek();
    try std.testing.expect(peeked != null);
    try std.testing.expectEqual(@as(f32, 0.3), peeked.?.value);

    // Item should still be there
    const popped = ring.pop();
    try std.testing.expect(popped != null);
    try std.testing.expectEqual(@as(f32, 0.3), popped.?.value);

    // Now empty
    try std.testing.expect(ring.pop() == null);
}

test "RingBuffer len tracks count" {
    var ring = RingBuffer(Signal, 16){};

    try std.testing.expectEqual(@as(usize, 0), ring.len());

    _ = ring.push(Signal.now(.cloud, .arousal, 0.1));
    try std.testing.expectEqual(@as(usize, 1), ring.len());

    _ = ring.push(Signal.now(.cloud, .arousal, 0.2));
    try std.testing.expectEqual(@as(usize, 2), ring.len());

    _ = ring.pop();
    try std.testing.expectEqual(@as(usize, 1), ring.len());
}

test "RingBuffer handles wraparound" {
    var ring = RingBuffer(Signal, 4){};

    // Fill buffer
    for (0..4) |i| {
        const val: f32 = @floatFromInt(i);
        try std.testing.expect(ring.push(Signal.now(.cloud, .arousal, val / 10.0)));
    }

    // Buffer full
    try std.testing.expect(!ring.push(Signal.now(.cloud, .arousal, 0.9)));

    // Pop all
    for (0..4) |i| {
        const expected: f32 = @floatFromInt(i);
        const popped = ring.pop();
        try std.testing.expect(popped != null);
        try std.testing.expectApproxEqAbs(expected / 10.0, popped.?.value, 0.001);
    }

    // Empty again
    try std.testing.expect(ring.pop() == null);

    // Can push again (wraparound)
    try std.testing.expect(ring.push(Signal.now(.cloud, .arousal, 0.99)));
    try std.testing.expectEqual(@as(usize, 1), ring.len());
}

test "RingBuffer full returns false on push" {
    var ring = RingBuffer(Signal, 4){};

    // Fill
    for (0..4) |_| {
        try std.testing.expect(ring.push(Signal.now(.cloud, .arousal, 0.5)));
    }

    // Full - should return false
    try std.testing.expect(!ring.push(Signal.now(.cloud, .arousal, 0.5)));
}

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS NERVE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "VagusNerve init" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    try std.testing.expectEqual(@as(u64, 0), nerve.signals_sent);
    try std.testing.expectEqual(@as(u64, 0), nerve.signals_received);
}

test "VagusNerve send updates state" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Initial arousal
    try std.testing.expectEqual(@as(f32, 0.3), state.arousal);

    // Send arousal signal
    const sig = Signal.now(.cloud, .arousal, 0.9);
    try std.testing.expect(nerve.send(sig));

    // State should be updated
    try std.testing.expectEqual(@as(f32, 0.9), state.arousal);
    try std.testing.expectEqual(@as(u64, 1), nerve.signals_sent);
}

test "VagusNerve send all signal types" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Emotional
    _ = nerve.send(Signal.now(.cloud, .arousal, 0.1));
    try std.testing.expectEqual(@as(f32, 0.1), state.arousal);

    _ = nerve.send(Signal.now(.cloud, .valence, -0.5));
    try std.testing.expectEqual(@as(f32, -0.5), state.valence);

    _ = nerve.send(Signal.now(.cloud, .warmth, 0.7));
    try std.testing.expectEqual(@as(f32, 0.7), state.chamber_warmth);

    _ = nerve.send(Signal.now(.cloud, .void_level, 0.3));
    try std.testing.expectEqual(@as(f32, 0.3), state.chamber_void);

    _ = nerve.send(Signal.now(.cloud, .tension, 0.6));
    try std.testing.expectEqual(@as(f32, 0.6), state.chamber_tension);

    _ = nerve.send(Signal.now(.cloud, .sacred, 0.4));
    try std.testing.expectEqual(@as(f32, 0.4), state.chamber_sacred);

    // Cognitive
    _ = nerve.send(Signal.now(.arianna, .coherence, 0.9));
    try std.testing.expectEqual(@as(f32, 0.9), state.coherence);

    _ = nerve.send(Signal.now(.arianna, .entropy, 0.2));
    try std.testing.expectEqual(@as(f32, 0.2), state.entropy);

    // Trauma
    _ = nerve.send(Signal.now(.inner_world, .trauma, 0.5));
    try std.testing.expectEqual(@as(f32, 0.5), state.trauma_level);

    // Temporal
    _ = nerve.send(Signal.now(.inner_world, .prophecy_debt, 0.8));
    try std.testing.expectEqual(@as(f32, 0.8), state.prophecy_debt);

    _ = nerve.send(Signal.now(.inner_world, .drift_direction, -0.3));
    try std.testing.expectEqual(@as(f32, -0.3), state.drift_direction);

    _ = nerve.send(Signal.now(.inner_world, .drift_speed, 0.4));
    try std.testing.expectEqual(@as(f32, 0.4), state.drift_speed);

    // Memory
    _ = nerve.send(Signal.now(.limpha, .memory_pressure, 0.6));
    try std.testing.expectEqual(@as(f32, 0.6), state.memory_pressure);
}

test "VagusNerve receive" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Send
    _ = nerve.send(Signal.now(.cloud, .arousal, 0.5));

    // Receive
    const received = nerve.receive();
    try std.testing.expect(received != null);
    try std.testing.expectEqual(@as(f32, 0.5), received.?.value);
    try std.testing.expectEqual(@as(u64, 1), nerve.signals_received);
}

test "VagusNerve heartbeat" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // First tick should generate heartbeat
    nerve.heartbeat_interval_us = 0; // Immediate
    const beat = nerve.tick();

    try std.testing.expect(beat != null);
    try std.testing.expectEqual(SignalType.heartbeat, beat.?.signal_type);
    try std.testing.expect(state.last_heartbeat_us > 0);
}

test "VagusNerve snapshot" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    state.arousal = 0.7;
    state.trauma_level = 0.3;

    const snap = nerve.snapshot();

    try std.testing.expectEqual(@as(f32, 0.7), snap.arousal);
    try std.testing.expectEqual(@as(f32, 0.3), snap.trauma_level);
}

test "VagusNerve update_count increments" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    try std.testing.expectEqual(@as(u64, 0), state.update_count);

    _ = nerve.send(Signal.now(.cloud, .arousal, 0.5));
    try std.testing.expectEqual(@as(u64, 1), state.update_count);

    _ = nerve.send(Signal.now(.cloud, .valence, 0.3));
    try std.testing.expectEqual(@as(u64, 2), state.update_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CROSSFIRE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "CrossFire coherence high for balanced chambers" {
    const balanced: [6]f32 = .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    const coh = CrossFireMatrix.coherence(balanced);

    // Balanced = high coherence
    try std.testing.expect(coh > 0.9);
}

test "CrossFire coherence low for unbalanced chambers" {
    const unbalanced: [6]f32 = .{ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
    const coh = CrossFireMatrix.coherence(unbalanced);

    // Unbalanced = low coherence
    try std.testing.expect(coh < 0.5);
}

test "CrossFire coherence zero for extreme spread" {
    const extreme: [6]f32 = .{ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
    const coh = CrossFireMatrix.coherence(extreme);

    try std.testing.expect(coh < 0.1);
}

test "CrossFire apply modifies chambers" {
    var matrix = CrossFireMatrix{};
    const input: [6]f32 = .{ 0.8, 0.2, 0.3, 0.5, 0.6, 0.4 };

    const output = matrix.apply(input);

    // Output should be different from input
    var different = false;
    for (0..6) |i| {
        if (@abs(output[i] - input[i]) > 0.001) {
            different = true;
            break;
        }
    }
    try std.testing.expect(different);

    // Output should be clamped to [0, 1]
    for (output) |v| {
        try std.testing.expect(v >= 0.0);
        try std.testing.expect(v <= 1.0);
    }
}

test "CrossFire warmth suppresses void" {
    var matrix = CrossFireMatrix{};

    // High warmth
    const high_warmth: [6]f32 = .{ 0.9, 0.5, 0.3, 0.3, 0.5, 0.3 };
    const result = matrix.apply(high_warmth);

    // Void should decrease (warmth suppresses void)
    try std.testing.expect(result[1] < high_warmth[1]);
}

test "CrossFire void suppresses warmth" {
    var matrix = CrossFireMatrix{};

    // High void
    const high_void: [6]f32 = .{ 0.5, 0.9, 0.3, 0.3, 0.5, 0.3 };
    const result = matrix.apply(high_void);

    // Warmth should decrease (void suppresses warmth)
    try std.testing.expect(result[0] < high_void[0]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// C API TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "C API vagus_init" {
    const result = vagus.vagus_init();
    try std.testing.expectEqual(@as(c_int, 0), result);
}

test "C API vagus_send" {
    _ = vagus.vagus_init();

    const result = vagus.vagus_send(
        @intFromEnum(Source.cloud),
        @intFromEnum(SignalType.arousal),
        0.8,
    );
    try std.testing.expectEqual(@as(c_int, 0), result);

    // Check state was updated
    const arousal = vagus.vagus_get_arousal();
    try std.testing.expectEqual(@as(f32, 0.8), arousal);
}

test "C API vagus_get_state" {
    _ = vagus.vagus_init();

    const state = vagus.vagus_get_state();
    try std.testing.expect(state != null);
    try std.testing.expectEqual(@as(u32, 1), state.vagus_version);
}

test "C API vagus_get_chambers" {
    _ = vagus.vagus_init();

    // Set chambers via send
    _ = vagus.vagus_send(@intFromEnum(Source.cloud), @intFromEnum(SignalType.warmth), 0.7);
    _ = vagus.vagus_send(@intFromEnum(Source.cloud), @intFromEnum(SignalType.void_level), 0.3);

    var chambers: [6]f32 = undefined;
    vagus.vagus_get_chambers(&chambers);

    try std.testing.expectEqual(@as(f32, 0.7), chambers[0]); // warmth
    try std.testing.expectEqual(@as(f32, 0.3), chambers[1]); // void
}

test "C API vagus_tick" {
    _ = vagus.vagus_init();

    // Should not crash
    vagus.vagus_tick();
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "Integration: full signal flow" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Simulate Cloud 200K sending chambers
    _ = nerve.send(Signal.now(.cloud, .warmth, 0.8));
    _ = nerve.send(Signal.now(.cloud, .void_level, 0.1));
    _ = nerve.send(Signal.now(.cloud, .tension, 0.4));

    // Simulate inner_world sending trauma
    _ = nerve.send(Signal.now(.inner_world, .trauma, 0.3));
    _ = nerve.send(Signal.now(.inner_world, .prophecy_debt, 0.2));

    // Simulate Arianna sending coherence
    _ = nerve.send(Signal.now(.arianna, .coherence, 0.85));

    // Check all values propagated
    try std.testing.expectEqual(@as(f32, 0.8), state.chamber_warmth);
    try std.testing.expectEqual(@as(f32, 0.1), state.chamber_void);
    try std.testing.expectEqual(@as(f32, 0.4), state.chamber_tension);
    try std.testing.expectEqual(@as(f32, 0.3), state.trauma_level);
    try std.testing.expectEqual(@as(f32, 0.2), state.prophecy_debt);
    try std.testing.expectEqual(@as(f32, 0.85), state.coherence);

    // 6 signals sent
    try std.testing.expectEqual(@as(u64, 6), nerve.signals_sent);
    try std.testing.expectEqual(@as(u64, 6), state.update_count);
}

test "Integration: SARTRE reads full state" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Set up a complex state
    _ = nerve.send(Signal.now(.cloud, .arousal, 0.7));
    _ = nerve.send(Signal.now(.cloud, .valence, 0.3));
    _ = nerve.send(Signal.now(.inner_world, .trauma, 0.2));
    _ = nerve.send(Signal.now(.inner_world, .drift_direction, -0.4));
    _ = nerve.send(Signal.now(.arianna, .coherence, 0.9));

    // SARTRE reads snapshot
    const snap = nerve.snapshot();

    // Compute what SARTRE would perceive
    const warmth = snap.chamber_warmth;
    const void_level = snap.chamber_void;
    const pressure = snap.trauma_level * 0.4 + snap.memory_pressure * 0.3 + snap.prophecy_debt * 0.1;
    const flow = snap.coherence * 0.4 + (1.0 - snap.entropy) * 0.3;

    // Basic interoceptive readings
    try std.testing.expect(warmth >= 0.0 and warmth <= 1.0);
    try std.testing.expect(pressure >= 0.0);
    try std.testing.expect(flow >= 0.0);
}

test "Integration: concurrent-like access pattern" {
    var state = SharedState{};
    var nerve = VagusNerve.init(&state);

    // Simulate rapid updates from multiple sources
    for (0..100) |i| {
        const val: f32 = @as(f32, @floatFromInt(i % 10)) / 10.0;

        _ = nerve.send(Signal.now(.cloud, .arousal, val));
        _ = nerve.send(Signal.now(.inner_world, .trauma, val * 0.5));
        _ = nerve.send(Signal.now(.arianna, .coherence, 1.0 - val * 0.3));

        // Tick heartbeat
        _ = nerve.tick();
    }

    // Should have processed all signals
    try std.testing.expectEqual(@as(u64, 300), nerve.signals_sent);

    // State should be valid
    try std.testing.expect(state.arousal >= 0.0 and state.arousal <= 1.0);
    try std.testing.expect(state.trauma_level >= 0.0 and state.trauma_level <= 1.0);
    try std.testing.expect(state.coherence >= 0.0 and state.coherence <= 1.0);
}
