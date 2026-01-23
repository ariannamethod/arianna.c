/*
 * vagus_delta.h — Bridge between Nervous System and Learning
 *
 * Connects:
 *   Vagus (field state)  →  Delta (experience learning)
 *   Locus (resonance)    →  Microtrain (notorch plasticity)
 *
 * The nerve speaks. The delta listens. Learning happens.
 */

#ifndef VAGUS_DELTA_H
#define VAGUS_DELTA_H

#include "delta.h"
#include "../vagus/vagus.h"
#include "../locus/locus.h"

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE PATTERNS (from Locus)
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    RESONANCE_NONE = 0,
    RESONANCE_CRISIS,         // High arousal + low coherence + trauma
    RESONANCE_DISSOLUTION,    // High void + low warmth + memory pressure
    RESONANCE_EMERGENCE,      // High coherence + low entropy + prophecy
    RESONANCE_TRANSCENDENCE,  // High sacred + low tension + flow
    RESONANCE_GEOMETRY_SHIFT, // Delta exceeded threshold
} ResonancePattern;

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS-AWARE SHARD (captures field geometry)
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Core shard data
    ExperienceShard shard;

    // Vagus snapshot at creation time
    float arousal;
    float valence;
    float coherence;
    float entropy;
    float chambers[6];  // warmth, void, tension, sacred, flow, complex
    float trauma_level;
    float prophecy_debt;
    float memory_pressure;

    // Locus geometry at creation
    float geometry_pressure;
    float geometry_flow;
    float geometry_depth;
    ResonancePattern trigger_pattern;

    // Metadata
    uint64_t created_us;
    uint32_t training_cycles;
    float total_signal;
} VagusAwareShard;

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE-TRIGGERED TRAINER
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Core components
    MicroTrainer trainer;
    ExperienceAccumulator accumulator;
    Locus locus;

    // Active shard being trained
    VagusAwareShard* active_shard;

    // Training modulation
    float crisis_lr_boost;       // Learning rate multiplier in crisis
    float emergence_lr_boost;    // Learning rate multiplier in emergence
    float transcendence_freeze;  // Probability to freeze in transcendence

    // Pattern-specific behavior
    int crisis_mode;             // 1 = heightened learning
    int emergence_mode;          // 1 = consolidation phase
    int dissolution_mode;        // 1 = protective mode (reduce learning)

    // Stats
    uint64_t total_resonances;
    uint64_t crisis_triggers;
    uint64_t emergence_triggers;
    uint64_t transcendence_triggers;
    uint64_t dissolution_triggers;

    // Vagus connection
    volatile VagusSharedState* vagus_state;
} ResonanceTrainer;

// ═══════════════════════════════════════════════════════════════════════════════
// CONVERSION: Vagus → Delta Signals
// ═══════════════════════════════════════════════════════════════════════════════

// Convert VagusSharedState to Delta Signals
void vagus_to_signals(const VagusSharedState* vagus, Signals* signals);

// Convert Signals back to Vagus updates
void signals_to_vagus(const Signals* signals, VagusSharedState* vagus);

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS-AWARE SHARD API
// ═══════════════════════════════════════════════════════════════════════════════

// Create new shard with current Vagus state snapshot
VagusAwareShard* create_vagus_shard(const char* name, int n_layers, int dim,
                                     const VagusSharedState* vagus,
                                     const Locus* locus);

// Free shard
void free_vagus_shard(VagusAwareShard* shard);

// Save shard with Vagus metadata
int save_vagus_shard(const VagusAwareShard* shard, const char* path);

// Load shard with Vagus metadata
VagusAwareShard* load_vagus_shard(const char* path, int n_layers, int dim);

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE TRAINER API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize resonance trainer
void init_resonance_trainer(ResonanceTrainer* rt, int dim, int vocab_size,
                            volatile VagusSharedState* vagus_state);

// Free resonance trainer
void free_resonance_trainer(ResonanceTrainer* rt);

// Tick — check Locus, maybe trigger training
// Returns resonance pattern if triggered, RESONANCE_NONE otherwise
ResonancePattern resonance_trainer_tick(ResonanceTrainer* rt);

// Experience step with resonance modulation
// Automatically adjusts learning based on Locus pattern
void resonance_experience_step(ResonanceTrainer* rt, LowRankDelta* delta,
                               const float* x, const float* probs,
                               int target_id);

// Force resonance check (useful for debugging)
ResonancePattern check_resonance(ResonanceTrainer* rt);

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING RATE MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

// Compute learning rate based on current resonance pattern
float compute_resonance_lr(ResonanceTrainer* rt, float base_lr);

// Compute signal strength based on field geometry
float compute_geometry_signal(const VagusSharedState* vagus);

// ═══════════════════════════════════════════════════════════════════════════════
// CALLBACKS (for integration)
// ═══════════════════════════════════════════════════════════════════════════════

// Callback type for resonance events
typedef void (*ResonanceCallback)(ResonancePattern pattern, void* ctx);

// Set callback for when resonance triggers
void set_resonance_callback(ResonanceTrainer* rt, ResonanceCallback cb, void* ctx);

#endif // VAGUS_DELTA_H
