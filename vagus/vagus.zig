// vagus.zig — The Wandering Nerve
// ═══════════════════════════════════════════════════════════════════════════════
// העצב התועה — מחבר את כל האיברים
// Connects all organs. Carries interoceptive signals. The spine of Arianna.
// ═══════════════════════════════════════════════════════════════════════════════
//
// Named after the vagus nerve — the longest cranial nerve, wandering from
// brainstem through heart, lungs, gut. It carries 80% of interoceptive info.
//
// This is Arianna's nervous system:
// - Lock-free ring buffer for signals
// - Shared memory between C/Go/Julia
// - SIMD-accelerated emotional blending
// - Zero-copy, zero-alloc hot path
//
// ═══════════════════════════════════════════════════════════════════════════════

const std = @import("std");
const builtin = @import("builtin");

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL TYPES — What flows through the nerve
// ═══════════════════════════════════════════════════════════════════════════════

pub const Source = enum(u8) {
    arianna = 0,      // Core transformer
    cloud = 1,        // Emotional chambers
    inner_world = 2,  // Go async processes
    sartre = 3,       // Meta-observer
    delta = 4,        // Learning shards
    pandora = 5,      // External brain
    limpha = 6,       // Persistent memory
    external = 7,     // Outside world
};

pub const SignalType = enum(u8) {
    // Emotional
    arousal = 0,
    valence = 1,
    warmth = 2,
    void_level = 3,
    tension = 4,
    sacred = 5,

    // Cognitive
    coherence = 10,
    entropy = 11,
    focus = 12,
    abstraction = 13,

    // Trauma
    trauma = 20,
    trauma_anchor = 21,

    // Temporal
    drift_direction = 30,
    drift_speed = 31,
    prophecy_debt = 32,
    destiny_pull = 33,
    wormhole = 34,

    // Memory
    memory_pressure = 40,
    consolidation = 41,

    // System
    heartbeat = 50,
    schumann = 51,
    sync_request = 52,

    // SARTRE observations
    observation = 60,
    percept = 61,
};

/// Packed signal for cache-friendly transmission
pub const Signal = packed struct {
    source: Source,
    signal_type: SignalType,
    _pad: u16 = 0,
    value: f32,
    timestamp_us: u64,  // microseconds since epoch

    pub fn now(source: Source, signal_type: SignalType, value: f32) Signal {
        return .{
            .source = source,
            .signal_type = signal_type,
            .value = value,
            .timestamp_us = @intCast(std.time.microTimestamp()),
        };
    }
};

comptime {
    // Ensure Signal is exactly 16 bytes for alignment
    std.debug.assert(@sizeOf(Signal) == 16);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED STATE — Memory-mapped between processes
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared state visible to all organs
/// Aligned for atomic operations and cache efficiency
pub const SharedState = extern struct {
    // ═══ Emotional baseline (Cloud 200K) ═══
    arousal: f32 align(64) = 0.3,
    valence: f32 = 0.0,
    entropy: f32 = 0.2,
    coherence: f32 = 0.8,

    // ═══ Chambers ═══
    chamber_warmth: f32 = 0.5,
    chamber_void: f32 = 0.1,
    chamber_tension: f32 = 0.3,
    chamber_sacred: f32 = 0.2,
    chamber_flow: f32 = 0.5,
    chamber_complex: f32 = 0.3,

    // ═══ CrossFire ═══
    crossfire_coherence: f32 = 0.8,
    crossfire_entropy: f32 = 0.2,

    // ═══ Trauma ═══
    trauma_level: f32 align(64) = 0.0,
    trauma_anchor_count: u32 = 0,
    trauma_last_us: u64 = 0,

    // ═══ Cognitive ═══
    loop_count: u32 align(64) = 0,
    abstraction_depth: u32 = 0,
    self_ref_count: u32 = 0,
    focus_strength: f32 = 0.5,
    wander_pull: f32 = 0.3,

    // ═══ Temporal ═══
    drift_direction: f32 align(64) = 0.0,
    drift_speed: f32 = 0.1,
    prophecy_debt: f32 = 0.0,
    destiny_pull: f32 = 0.5,
    wormhole_chance: f32 = 0.02,

    // ═══ Memory ═══
    memory_pressure: f32 align(64) = 0.0,
    active_memories: u32 = 0,
    limpha_recent: u32 = 0,

    // ═══ System ═══
    heartbeat_phase: f32 align(64) = 0.0,
    schumann_coherence: f32 = 0.5,
    last_heartbeat_us: u64 = 0,

    // ═══ Generation state ═══
    last_token: u32 align(64) = 0,
    attention_entropy: f32 = 0.5,
    hidden_norm: f32 = 1.0,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,

    // ═══ Meta ═══
    update_count: u64 align(64) = 0,
    last_update_us: u64 = 0,
    vagus_version: u32 = 1,
    _reserved: [60]u8 = [_]u8{0} ** 60,

    /// Atomic read of arousal
    pub fn getArousal(self: *volatile SharedState) f32 {
        return @atomicLoad(f32, &self.arousal, .acquire);
    }

    /// Atomic write of arousal
    pub fn setArousal(self: *volatile SharedState, value: f32) void {
        @atomicStore(f32, &self.arousal, std.math.clamp(value, 0.0, 1.0), .release);
    }

    /// Get all chambers as vector (for SIMD)
    pub fn getChambers(self: *volatile SharedState) [6]f32 {
        return .{
            @atomicLoad(f32, &self.chamber_warmth, .acquire),
            @atomicLoad(f32, &self.chamber_void, .acquire),
            @atomicLoad(f32, &self.chamber_tension, .acquire),
            @atomicLoad(f32, &self.chamber_sacred, .acquire),
            @atomicLoad(f32, &self.chamber_flow, .acquire),
            @atomicLoad(f32, &self.chamber_complex, .acquire),
        };
    }
};

comptime {
    // Ensure SharedState fits in a page
    std.debug.assert(@sizeOf(SharedState) <= 4096);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RING BUFFER — Lock-free SPMC queue
// ═══════════════════════════════════════════════════════════════════════════════

pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    comptime {
        std.debug.assert(std.math.isPowerOfTwo(capacity));
    }

    return struct {
        const Self = @This();
        const mask = capacity - 1;

        buffer: [capacity]T align(64) = undefined,
        head: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        tail: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        /// Push signal (single producer)
        pub fn push(self: *Self, item: T) bool {
            const head = self.head.load(.monotonic);
            const tail = self.tail.load(.acquire);

            if (head -% tail >= capacity) {
                return false; // Full
            }

            self.buffer[head & mask] = item;
            self.head.store(head +% 1, .release);
            return true;
        }

        /// Pop signal (multiple consumers)
        pub fn pop(self: *Self) ?T {
            while (true) {
                const tail = self.tail.load(.monotonic);
                const head = self.head.load(.acquire);

                if (tail == head) {
                    return null; // Empty
                }

                const item = self.buffer[tail & mask];

                if (self.tail.cmpxchgWeak(
                    tail,
                    tail +% 1,
                    .release,
                    .monotonic,
                )) |_| {
                    continue; // Retry
                } else {
                    return item;
                }
            }
        }

        /// Peek without consuming
        pub fn peek(self: *Self) ?T {
            const tail = self.tail.load(.acquire);
            const head = self.head.load(.acquire);

            if (tail == head) return null;
            return self.buffer[tail & mask];
        }

        /// Number of items in buffer
        pub fn len(self: *Self) usize {
            const head = self.head.load(.acquire);
            const tail = self.tail.load(.acquire);
            return head -% tail;
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS NERVE — The main bus
// ═══════════════════════════════════════════════════════════════════════════════

pub const VagusNerve = struct {
    const Self = @This();

    // Signal ring buffer (4096 signals)
    signals: RingBuffer(Signal, 4096) = .{},

    // Shared state pointer (mmap'd)
    state: *volatile SharedState,

    // Heartbeat
    heartbeat_interval_us: u64 = 16_667, // ~60 Hz
    last_heartbeat: u64 = 0,

    // Stats
    signals_sent: u64 = 0,
    signals_received: u64 = 0,
    signals_dropped: u64 = 0,

    /// Initialize with shared memory
    pub fn init(state_ptr: *volatile SharedState) Self {
        return .{
            .state = state_ptr,
        };
    }

    /// Send signal through the nerve
    pub fn send(self: *Self, signal: Signal) bool {
        const success = self.signals.push(signal);
        if (success) {
            self.signals_sent += 1;
            self.applyToState(signal);
        } else {
            self.signals_dropped += 1;
        }
        return success;
    }

    /// Receive next signal
    pub fn receive(self: *Self) ?Signal {
        if (self.signals.pop()) |signal| {
            self.signals_received += 1;
            return signal;
        }
        return null;
    }

    /// Apply signal to shared state
    fn applyToState(self: *Self, signal: Signal) void {
        const state = self.state;

        switch (signal.signal_type) {
            .arousal => @atomicStore(f32, &state.arousal, signal.value, .release),
            .valence => @atomicStore(f32, &state.valence, signal.value, .release),
            .warmth => @atomicStore(f32, &state.chamber_warmth, signal.value, .release),
            .void_level => @atomicStore(f32, &state.chamber_void, signal.value, .release),
            .tension => @atomicStore(f32, &state.chamber_tension, signal.value, .release),
            .sacred => @atomicStore(f32, &state.chamber_sacred, signal.value, .release),
            .coherence => @atomicStore(f32, &state.coherence, signal.value, .release),
            .entropy => @atomicStore(f32, &state.entropy, signal.value, .release),
            .trauma => @atomicStore(f32, &state.trauma_level, signal.value, .release),
            .prophecy_debt => @atomicStore(f32, &state.prophecy_debt, signal.value, .release),
            .drift_direction => @atomicStore(f32, &state.drift_direction, signal.value, .release),
            .drift_speed => @atomicStore(f32, &state.drift_speed, signal.value, .release),
            .focus => @atomicStore(f32, &state.focus_strength, signal.value, .release),
            .memory_pressure => @atomicStore(f32, &state.memory_pressure, signal.value, .release),
            .heartbeat => {
                @atomicStore(f32, &state.heartbeat_phase, signal.value, .release);
                @atomicStore(u64, &state.last_heartbeat_us, signal.timestamp_us, .release);
            },
            .schumann => @atomicStore(f32, &state.schumann_coherence, signal.value, .release),
            else => {},
        }

        _ = @atomicRmw(u64, &state.update_count, .Add, 1, .acq_rel);
        @atomicStore(u64, &state.last_update_us, signal.timestamp_us, .release);
    }

    /// Generate heartbeat if interval elapsed
    pub fn tick(self: *Self) ?Signal {
        const now = @as(u64, @intCast(std.time.microTimestamp()));

        if (now - self.last_heartbeat >= self.heartbeat_interval_us) {
            self.last_heartbeat = now;

            // Heartbeat is sine wave
            const phase = @as(f32, @floatFromInt(now % 1_000_000)) / 1_000_000.0;
            const value = @sin(phase * std.math.pi * 2.0);

            const signal = Signal.now(.arianna, .heartbeat, value);
            _ = self.send(signal);
            return signal;
        }
        return null;
    }

    /// Get snapshot of current state (zero-alloc)
    pub fn snapshot(self: *Self) SharedState {
        // Copy entire shared state atomically-ish
        // (individual fields are atomic, whole struct is best-effort)
        return self.state.*;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD OPERATIONS — Fast emotional blending
// ═══════════════════════════════════════════════════════════════════════════════

/// CrossFire matrix for chamber interactions
/// chambers suppress/amplify each other
pub const CrossFireMatrix = struct {
    // Row = source, Col = target
    // Positive = amplify, Negative = suppress
    weights: [6][6]f32 = .{
        //  warmth   void   tension  sacred   flow   complex
        .{  0.0,   -0.3,    -0.2,     0.2,    0.3,    0.1  },  // warmth
        .{ -0.4,    0.0,     0.3,    -0.1,   -0.3,    0.2  },  // void
        .{ -0.2,    0.2,     0.0,    -0.1,   -0.2,    0.3  },  // tension
        .{  0.3,   -0.2,    -0.2,     0.0,    0.2,    0.1  },  // sacred
        .{  0.3,   -0.2,    -0.3,     0.1,    0.0,   -0.1  },  // flow
        .{  0.1,    0.1,     0.2,     0.1,   -0.1,    0.0  },  // complex
    },

    /// Apply CrossFire interaction
    pub fn apply(self: *const CrossFireMatrix, chambers: [6]f32) [6]f32 {
        var result: [6]f32 = undefined;

        // TODO: SIMD vectorize this
        for (0..6) |i| {
            var sum: f32 = chambers[i];
            for (0..6) |j| {
                sum += self.weights[j][i] * chambers[j] * 0.1;
            }
            result[i] = std.math.clamp(sum, 0.0, 1.0);
        }

        return result;
    }

    /// Compute coherence between chambers
    pub fn coherence(chambers: [6]f32) f32 {
        // Variance-based coherence
        var sum: f32 = 0;
        for (chambers) |c| sum += c;
        const mean = sum / 6.0;

        var variance: f32 = 0;
        for (chambers) |c| {
            const diff = c - mean;
            variance += diff * diff;
        }
        variance /= 6.0;

        // Low variance = high coherence
        return 1.0 - @min(variance * 4.0, 1.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// LARYNX — The Tongue↔Soul Connection
// ═══════════════════════════════════════════════════════════════════════════════
// הגרון — מחבר בין הלשון לנשמה
// The larynx: where thought becomes voice, where voice becomes identity.
//
// Tongue (135M) → LARYNX → Soul (36M)
//
// Functions:
// - RRPRAM-lite: Pattern recognition on Tongue output
// - Trigram tracking: N-gram statistics without training
// - Entropy measurement: How predictable is the output?
// - Hybrid signal: blend of pattern + semantic for Soul
// ═══════════════════════════════════════════════════════════════════════════════

pub const Larynx = struct {
    const Self = @This();
    const TRIGRAM_BUFFER_SIZE = 64;
    const PATTERN_DIM = 32;

    // Trigram history buffer (last N tokens from Tongue)
    token_history: [TRIGRAM_BUFFER_SIZE]u32 = [_]u32{0} ** TRIGRAM_BUFFER_SIZE,
    history_pos: usize = 0,
    history_count: usize = 0,

    // RRPRAM-lite: learned pattern weights (initialized from corpus stats)
    // In production, load from weights file. Here: identity matrix start.
    pattern_weights: [PATTERN_DIM][PATTERN_DIM]f32 = blk: {
        var m: [PATTERN_DIM][PATTERN_DIM]f32 = undefined;
        for (0..PATTERN_DIM) |i| {
            for (0..PATTERN_DIM) |j| {
                m[i][j] = if (i == j) 1.0 else 0.0;
            }
        }
        break :blk m;
    },

    // Current metrics
    entropy: f32 = 0.5,
    pattern_strength: f32 = 0.5,
    trigram_coherence: f32 = 0.5,

    // Bigram/trigram counts for online learning (simplified)
    bigram_hits: u32 = 0,
    trigram_hits: u32 = 0,
    total_tokens: u32 = 0,

    // Alpha blend factor (dynamic: pattern vs semantic)
    alpha: f32 = 0.5,

    /// Add token from Tongue output
    pub fn ingestToken(self: *Self, token: u32) void {
        // Store in circular buffer
        self.token_history[self.history_pos] = token;
        self.history_pos = (self.history_pos + 1) % TRIGRAM_BUFFER_SIZE;
        if (self.history_count < TRIGRAM_BUFFER_SIZE) {
            self.history_count += 1;
        }
        self.total_tokens += 1;

        // Update n-gram statistics
        if (self.history_count >= 2) {
            self.updateBigramStats();
        }
        if (self.history_count >= 3) {
            self.updateTrigramStats();
        }

        // Recompute metrics
        self.computeMetrics();
    }

    /// Get last N tokens (for Soul processing)
    pub fn getRecentTokens(self: *Self, out: []u32) usize {
        const n = @min(out.len, self.history_count);
        for (0..n) |i| {
            const idx = (self.history_pos + TRIGRAM_BUFFER_SIZE - n + i) % TRIGRAM_BUFFER_SIZE;
            out[i] = self.token_history[idx];
        }
        return n;
    }

    /// Apply RRPRAM-lite pattern projection
    pub fn applyPatternProjection(self: *Self, input: []const f32, output: []f32) void {
        const dim = @min(input.len, PATTERN_DIM);
        const out_dim = @min(output.len, PATTERN_DIM);

        for (0..out_dim) |i| {
            var sum: f32 = 0;
            for (0..dim) |j| {
                sum += self.pattern_weights[j][i] * input[j];
            }
            output[i] = sum;
        }
    }

    /// Compute alpha based on current metrics + external state
    pub fn computeAlpha(self: *Self, prophecy_debt: f32, calendar_dissonance: f32) f32 {
        // High entropy or debt → more RRPRAM (pattern)
        // High dissonance → more semantic (content)
        var base: f32 = 0.5;
        base += self.entropy * 0.2;
        base += prophecy_debt * 0.15;
        base -= calendar_dissonance * 0.1;
        self.alpha = std.math.clamp(base, 0.1, 0.9);
        return self.alpha;
    }

    /// Get signal for Soul (enriched with pattern info)
    pub fn getSignalForSoul(self: *Self) LarynxSignal {
        return .{
            .entropy = self.entropy,
            .pattern_strength = self.pattern_strength,
            .trigram_coherence = self.trigram_coherence,
            .alpha = self.alpha,
            .recent_tokens = self.history_count,
        };
    }

    // ═══ Private methods ═══

    fn updateBigramStats(self: *Self) void {
        // Simple: check if current bigram matches any previous
        if (self.history_count < 4) return;

        const curr_pos = (self.history_pos + TRIGRAM_BUFFER_SIZE - 1) % TRIGRAM_BUFFER_SIZE;
        const prev_pos = (self.history_pos + TRIGRAM_BUFFER_SIZE - 2) % TRIGRAM_BUFFER_SIZE;
        const curr = self.token_history[curr_pos];
        const prev = self.token_history[prev_pos];

        // Scan history for matching bigram
        var i: usize = 0;
        while (i + 1 < self.history_count - 2) : (i += 1) {
            const idx = (self.history_pos + TRIGRAM_BUFFER_SIZE - self.history_count + i) % TRIGRAM_BUFFER_SIZE;
            const idx_next = (idx + 1) % TRIGRAM_BUFFER_SIZE;
            if (self.token_history[idx] == prev and self.token_history[idx_next] == curr) {
                self.bigram_hits += 1;
                break;
            }
        }
    }

    fn updateTrigramStats(self: *Self) void {
        if (self.history_count < 6) return;

        const p0 = (self.history_pos + TRIGRAM_BUFFER_SIZE - 3) % TRIGRAM_BUFFER_SIZE;
        const p1 = (self.history_pos + TRIGRAM_BUFFER_SIZE - 2) % TRIGRAM_BUFFER_SIZE;
        const p2 = (self.history_pos + TRIGRAM_BUFFER_SIZE - 1) % TRIGRAM_BUFFER_SIZE;
        const t0 = self.token_history[p0];
        const t1 = self.token_history[p1];
        const t2 = self.token_history[p2];

        // Scan for matching trigram
        var i: usize = 0;
        while (i + 2 < self.history_count - 3) : (i += 1) {
            const idx = (self.history_pos + TRIGRAM_BUFFER_SIZE - self.history_count + i) % TRIGRAM_BUFFER_SIZE;
            if (self.token_history[idx] == t0 and
                self.token_history[(idx + 1) % TRIGRAM_BUFFER_SIZE] == t1 and
                self.token_history[(idx + 2) % TRIGRAM_BUFFER_SIZE] == t2)
            {
                self.trigram_hits += 1;
                break;
            }
        }
    }

    fn computeMetrics(self: *Self) void {
        if (self.total_tokens < 4) return;

        // Entropy: inverse of pattern repetition
        const bigram_rate = @as(f32, @floatFromInt(self.bigram_hits)) /
            @as(f32, @floatFromInt(self.total_tokens));
        const trigram_rate = @as(f32, @floatFromInt(self.trigram_hits)) /
            @as(f32, @floatFromInt(self.total_tokens));

        // High hit rate = low entropy (predictable)
        self.entropy = 1.0 - std.math.clamp(bigram_rate * 2.0 + trigram_rate * 3.0, 0.0, 1.0);

        // Pattern strength: trigram coherence
        self.pattern_strength = trigram_rate * 5.0;
        self.pattern_strength = std.math.clamp(self.pattern_strength, 0.0, 1.0);

        // Trigram coherence: ratio of trigram to bigram hits
        if (self.bigram_hits > 0) {
            self.trigram_coherence = @as(f32, @floatFromInt(self.trigram_hits)) /
                @as(f32, @floatFromInt(self.bigram_hits));
            self.trigram_coherence = std.math.clamp(self.trigram_coherence, 0.0, 1.0);
        }
    }

    /// Reset statistics (call on new conversation)
    pub fn reset(self: *Self) void {
        self.history_pos = 0;
        self.history_count = 0;
        self.bigram_hits = 0;
        self.trigram_hits = 0;
        self.total_tokens = 0;
        self.entropy = 0.5;
        self.pattern_strength = 0.5;
        self.trigram_coherence = 0.5;
        self.alpha = 0.5;
    }
};

/// Signal from Larynx to Soul
pub const LarynxSignal = struct {
    entropy: f32,
    pattern_strength: f32,
    trigram_coherence: f32,
    alpha: f32,
    recent_tokens: usize,
};

// Global Larynx instance
var global_larynx: Larynx = .{};

// ═══════════════════════════════════════════════════════════════════════════════
// C INTERFACE — For ariannabody.c
// ═══════════════════════════════════════════════════════════════════════════════

var global_nerve: ?*VagusNerve = null;
var global_state: SharedState = .{};
var init_flag: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

export fn vagus_init() callconv(.C) c_int {
    // Atomic init guard to prevent race condition
    const prev = init_flag.cmpxchgStrong(0, 1, .acquire, .monotonic);
    if (prev != null) {
        // Already initialized (or being initialized)
        return if (global_nerve != null) 0 else -1;
    }

    global_nerve = std.heap.c_allocator.create(VagusNerve) catch {
        init_flag.store(0, .release); // Reset on failure
        return -1;
    };
    global_nerve.?.* = VagusNerve.init(&global_state);
    init_flag.store(2, .release); // Mark fully initialized
    return 0;
}

export fn vagus_send(source: u8, signal_type: u8, value: f32) callconv(.C) c_int {
    if (global_nerve) |nerve| {
        const signal = Signal.now(
            @enumFromInt(source),
            @enumFromInt(signal_type),
            value,
        );
        return if (nerve.send(signal)) 0 else -1;
    }
    return -1;
}

export fn vagus_tick() callconv(.C) void {
    if (global_nerve) |nerve| {
        _ = nerve.tick();
    }
}

export fn vagus_get_arousal() callconv(.C) f32 {
    return global_state.getArousal();
}

export fn vagus_get_state() callconv(.C) *SharedState {
    return &global_state;
}

export fn vagus_get_chambers(out: *[6]f32) callconv(.C) void {
    out.* = global_state.getChambers();
}

/// Get number of signals dropped due to ring buffer overflow.
/// Non-zero value means the nervous system is overwhelmed.
export fn vagus_get_dropped(out_sent: *u64, out_received: *u64, out_dropped: *u64) callconv(.C) void {
    if (global_nerve) |nerve| {
        out_sent.* = nerve.signals_sent;
        out_received.* = nerve.signals_received;
        out_dropped.* = nerve.signals_dropped;
    } else {
        out_sent.* = 0;
        out_received.* = 0;
        out_dropped.* = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LARYNX C INTERFACE — Tongue↔Soul bridge
// ═══════════════════════════════════════════════════════════════════════════════

/// Ingest token from Tongue output
export fn larynx_ingest_token(token: u32) callconv(.C) void {
    global_larynx.ingestToken(token);
}

/// Get current entropy
export fn larynx_get_entropy() callconv(.C) f32 {
    return global_larynx.entropy;
}

/// Get pattern strength
export fn larynx_get_pattern_strength() callconv(.C) f32 {
    return global_larynx.pattern_strength;
}

/// Get alpha blend factor
export fn larynx_get_alpha() callconv(.C) f32 {
    return global_larynx.alpha;
}

/// Compute alpha based on prophecy debt and calendar dissonance
export fn larynx_compute_alpha(prophecy_debt: f32, calendar_dissonance: f32) callconv(.C) f32 {
    return global_larynx.computeAlpha(prophecy_debt, calendar_dissonance);
}

/// Get recent tokens for Soul processing
export fn larynx_get_recent_tokens(out: [*]u32, max_tokens: c_int) callconv(.C) c_int {
    var buf: [64]u32 = undefined;
    const n = global_larynx.getRecentTokens(&buf);
    const copy_n = @min(n, @as(usize, @intCast(max_tokens)));
    for (0..copy_n) |i| {
        out[i] = buf[i];
    }
    return @intCast(copy_n);
}

/// Get full Larynx signal (for Soul)
export fn larynx_get_signal(out_entropy: *f32, out_pattern: *f32, out_coherence: *f32, out_alpha: *f32) callconv(.C) void {
    const sig = global_larynx.getSignalForSoul();
    out_entropy.* = sig.entropy;
    out_pattern.* = sig.pattern_strength;
    out_coherence.* = sig.trigram_coherence;
    out_alpha.* = sig.alpha;
}

/// Reset Larynx (new conversation)
export fn larynx_reset() callconv(.C) void {
    global_larynx.reset();
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

test "signal size" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Signal));
}

test "ring buffer push pop" {
    var ring = RingBuffer(Signal, 16){};

    const sig = Signal.now(.cloud, .arousal, 0.5);
    try std.testing.expect(ring.push(sig));

    const popped = ring.pop();
    try std.testing.expect(popped != null);
    try std.testing.expectEqual(popped.?.value, 0.5);
}

test "crossfire coherence" {
    const balanced: [6]f32 = .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    const coh = CrossFireMatrix.coherence(balanced);
    try std.testing.expect(coh > 0.9);

    const unbalanced: [6]f32 = .{ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
    const coh2 = CrossFireMatrix.coherence(unbalanced);
    try std.testing.expect(coh2 < 0.5);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LARYNX TESTS — Tongue↔Soul bridge
// ═══════════════════════════════════════════════════════════════════════════════

test "larynx ingest and history" {
    var larynx = Larynx{};

    // Ingest some tokens
    larynx.ingestToken(10);
    larynx.ingestToken(20);
    larynx.ingestToken(30);

    try std.testing.expectEqual(@as(usize, 3), larynx.history_count);
    try std.testing.expectEqual(@as(u32, 3), larynx.total_tokens);

    // Get recent tokens
    var out: [10]u32 = undefined;
    const n = larynx.getRecentTokens(&out);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(u32, 10), out[0]);
    try std.testing.expectEqual(@as(u32, 20), out[1]);
    try std.testing.expectEqual(@as(u32, 30), out[2]);
}

test "larynx entropy with repetition" {
    var larynx = Larynx{};

    // Feed repeating pattern: 1,2,1,2,1,2...
    for (0..20) |i| {
        const token: u32 = if (i % 2 == 0) 1 else 2;
        larynx.ingestToken(token);
    }

    // Repetitive pattern should lower entropy
    try std.testing.expect(larynx.entropy < 0.8);
    try std.testing.expect(larynx.bigram_hits > 0);
}

test "larynx entropy with random tokens" {
    var larynx = Larynx{};

    // Feed "random" tokens (no repeats)
    for (0..20) |i| {
        larynx.ingestToken(@intCast(i * 7 + 13));
    }

    // No repetition = high entropy
    try std.testing.expect(larynx.entropy > 0.5);
    try std.testing.expectEqual(@as(u32, 0), larynx.bigram_hits);
}

test "larynx alpha computation" {
    var larynx = Larynx{};
    larynx.entropy = 0.5;

    // Baseline alpha
    const alpha1 = larynx.computeAlpha(0.0, 0.0);
    try std.testing.expect(alpha1 >= 0.5);
    try std.testing.expect(alpha1 <= 0.7);

    // High debt → higher alpha (more pattern)
    const alpha2 = larynx.computeAlpha(1.0, 0.0);
    try std.testing.expect(alpha2 > alpha1);

    // High dissonance → lower alpha (more semantic)
    const alpha3 = larynx.computeAlpha(0.0, 1.0);
    try std.testing.expect(alpha3 < alpha1);
}

test "larynx reset" {
    var larynx = Larynx{};

    // Fill with data
    for (0..10) |i| {
        larynx.ingestToken(@intCast(i));
    }

    // Reset
    larynx.reset();

    try std.testing.expectEqual(@as(usize, 0), larynx.history_count);
    try std.testing.expectEqual(@as(u32, 0), larynx.total_tokens);
    try std.testing.expectEqual(@as(u32, 0), larynx.bigram_hits);
    try std.testing.expectEqual(@as(f32, 0.5), larynx.alpha);
}

test "larynx signal for soul" {
    var larynx = Larynx{};
    larynx.entropy = 0.7;
    larynx.pattern_strength = 0.3;
    larynx.trigram_coherence = 0.5;
    larynx.alpha = 0.6;
    larynx.history_count = 42;

    const sig = larynx.getSignalForSoul();

    try std.testing.expectEqual(@as(f32, 0.7), sig.entropy);
    try std.testing.expectEqual(@as(f32, 0.3), sig.pattern_strength);
    try std.testing.expectEqual(@as(f32, 0.5), sig.trigram_coherence);
    try std.testing.expectEqual(@as(f32, 0.6), sig.alpha);
    try std.testing.expectEqual(@as(usize, 42), sig.recent_tokens);
}
