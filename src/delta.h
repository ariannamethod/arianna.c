/*
 * delta.h - Dynamic attention deltas for arianna.c
 *
 * Concept: Base personality + experience deltas
 * Deltas modify ATTENTION, not weights directly
 *
 * Hierarchy:
 *   1. Core personality (arianna.bin) - immutable, "who I am"
 *   2. Experience deltas (shards) - accumulated, "what I lived"
 *   3. Context deltas (runtime) - ephemeral, "what's happening now"
 */

#ifndef DELTA_H
#define DELTA_H

#include "arianna.h"

// ============================================================
// Configuration
// ============================================================

#define MAX_SHARDS 32           // Max experience shards loaded
#define DELTA_RANK 8            // LoRA-style low rank
#define MAX_SIGNALS 16          // Signal dimensions

// ============================================================
// Signals - extracted from context
// ============================================================

typedef struct {
    float arousal;              // Activation level [0,1]
    float entropy;              // Uncertainty/creativity [0,1]
    float tension;              // Conflict/pressure [0,1]
    float warmth;               // Emotional presence [0,1]
    float focus;                // Attention narrowness [0,1]
    float novelty;              // New vs familiar [0,1]
    float recursion_depth;      // Self-reference level [0,1]
    float resonance;            // Pattern matching strength [0,1]
} Signals;

// ============================================================
// Delta structures
// ============================================================

// Single low-rank delta: ΔW = A @ B
typedef struct {
    float* A;       // [out_dim, rank] - always float32
    float* B;       // [rank, in_dim]  - float32 (working copy for updates)
    int out_dim;
    int in_dim;
    int rank;

    // int8 quantization for B (inference-time compression)
    int8_t* B_quant;   // [rank, in_dim] int8 quantized B (NULL if not quantized)
    float B_scale;      // Scale factor: B_float ≈ B_quant * B_scale + B_zero
    float B_zero;       // Zero point
    int quantized;      // 1 if B_quant is valid and should be used for apply
} LowRankDelta;

// Experience shard - collection of deltas for all layers
typedef struct {
    char name[64];

    // Attention deltas per layer: modify Q,K,V,O projections
    LowRankDelta* attn_q_deltas;  // [n_layers]
    LowRankDelta* attn_k_deltas;  // [n_layers]
    LowRankDelta* attn_v_deltas;  // [n_layers]
    LowRankDelta* attn_o_deltas;  // [n_layers] - output projection

    // Metadata
    float strength;              // How much this shard influences
    int n_layers;
    int n_delta_types;           // 3 for old format (Q,K,V), 4 for new (Q,K,V,O)
} ExperienceShard;

// Delta bank - all loaded shards
typedef struct {
    ExperienceShard shards[MAX_SHARDS];
    int n_shards;

    // Current mix weights (from signals)
    float mix[MAX_SHARDS];

    // Combined delta cache (recomputed when mix changes)
    float* combined_q_delta;     // [n_layers, dim, dim]
    float* combined_k_delta;
    float* combined_v_delta;
    float* combined_o_delta;
    int cache_valid;
} DeltaBank;

// ============================================================
// Signal extraction
// ============================================================

// Extract signals from recent tokens/context
void extract_signals(Signals* sig, int* tokens, int n_tokens, float* hidden_states);

// Default neutral signals
void init_signals(Signals* sig);

// ============================================================
// Delta mixing
// ============================================================

// Compute mix weights from signals
void compute_mix(DeltaBank* bank, Signals* sig);

// Apply mixed delta to Q projection
void apply_q_delta(DeltaBank* bank, float* q, float* x, int layer, int dim);

// Apply mixed delta to K projection
void apply_k_delta(DeltaBank* bank, float* k, float* x, int layer, int dim);

// Apply mixed delta to V projection
void apply_v_delta(DeltaBank* bank, float* v, float* x, int layer, int dim);

// ============================================================
// Shard I/O
// ============================================================

// Load shard from binary file
int load_shard(ExperienceShard* shard, const char* path, int n_layers, int dim);

// Save shard to binary file
int save_shard(ExperienceShard* shard, const char* path);

// Free shard memory
void free_shard(ExperienceShard* shard);

// ============================================================
// Delta bank management
// ============================================================

void init_delta_bank(DeltaBank* bank);
void free_delta_bank(DeltaBank* bank);
int add_shard(DeltaBank* bank, const char* path, int n_layers, int dim);
int delta_bank_has_shard(DeltaBank* bank, const char* name);

// ============================================================
// Microtraining - online learning (notorch style from lang/lora.c)
// ============================================================

typedef struct {
    float learning_rate;
    float momentum;
    float decay;

    // Hebbian traces
    float* pre_trace;    // [dim]
    float* post_trace;   // [dim]

    // Contrastive learning params (from lora.c)
    float push;          // Strength of target boost
    float pull;          // Strength of competitor suppression
    int topk;            // How many competitors to suppress

    // Deterministic noise channel (from lora.c)
    unsigned int seed;
    float* u;            // [rank] inner experience channel
    float* dy;           // [out_dim] desired output delta

    // Dimensions
    int dim;
    int vocab_size;
} MicroTrainer;

void init_microtrainer(MicroTrainer* mt, int dim);
void free_microtrainer(MicroTrainer* mt);

// Update delta based on attention patterns (Hebbian)
void micro_update(MicroTrainer* mt, LowRankDelta* delta,
                  float* pre, float* post, float reward);

// ============================================================
// Notorch Plasticity (from lang/lora.c)
// "This is NOT gradient descent. It's plasticity."
// ============================================================

// Build dy from probs: push target, pull competitors
void build_dy_from_probs(MicroTrainer* mt, float* dy_out,
                         const float* probs, int vocab_size,
                         int target_id);

// Notorch step: plasticity without backprop
void notorch_step(MicroTrainer* mt, LowRankDelta* delta,
                  const float* x, const float* dy, float signal);

// Experience step: wrapper that builds dy and applies notorch
void experience_step(MicroTrainer* mt, LowRankDelta* delta,
                     const float* x, const float* probs,
                     int target_id, float signal);

// ============================================================
// Notorch Microlearning Revolution (5 improvements)
// "Pure C plasticity without PyTorch compromise"
// ============================================================

// 1. Resonance-Gated Plasticity: learn more when aligned with identity
void experience_step_gated(MicroTrainer* mt, LowRankDelta* delta,
                           const float* x, const float* probs,
                           int target_id, float signal,
                           const float* identity_embedding, int dim);

// 2. Adaptive push/pull based on confidence
void set_adaptive_push_pull(MicroTrainer* mt, const float* probs,
                            int vocab_size, int target_id);

// 3. Quality-weighted signal from BodySense
float compute_quality_weight(float quality, float stuck, float boredom);

// 4. Check if channel should be frozen (spectral freezing)
int should_freeze_channel(LowRankDelta* delta, int channel, float threshold);

// 5. Consolidate crystallized channels into core experience
void consolidate_experience(LowRankDelta* delta, LowRankDelta* core,
                            int* frozen_mask, int n_frozen);

// Soft reset: gradual forgetting (scale down instead of zeroing)
void soft_reset_delta(LowRankDelta* delta, float keep_ratio);

// Clamp delta to prevent weight explosion
void clamp_delta(LowRankDelta* delta, float max_norm);

// Get delta norm (for monitoring)
float get_delta_norm(LowRankDelta* delta);

// int8 quantization for B matrices (inference-time compression)
void quantize_B(LowRankDelta* delta);
void dequantize_B_into(LowRankDelta* delta, float* out);  // out = [rank, in_dim]

// ============================================================
// Quantum Accumulation (Stanley-style)
// "Don't train on every token - accumulate until critical mass"
// ============================================================

#define ACCUM_BUFFER_SIZE 256  // Max accumulated experiences before forced flush

#include <pthread.h>

typedef struct {
    // Online summary stats (replaces full x_buffer)
    float* mean_x;                // [dim] - running mean of input activations
    int* target_counts;           // [vocab_size] - frequency of each target token
    float signal_sum;             // Sum of learning signals
    int experience_count;         // Number of accumulated experiences

    // Accumulation metrics (Stanley-style)
    float bytes_delta;            // Volume of new experience
    float resonance_mass;         // Weighted contextual relevance
    float novelty_mass;           // Distribution drift from baseline

    // Thresholds for triggering training
    float bytes_threshold;        // Trigger when bytes_delta exceeds this
    float resonance_threshold;    // Trigger when resonance_mass exceeds this
    float novelty_threshold;      // Trigger when novelty_mass exceeds this

    // Cooldown (minimum time between training cycles)
    float cooldown_remaining;     // Seconds until can train again
    float cooldown_period;        // Minimum gap between training

    // Dimensions (for allocation)
    int dim;
    int vocab_size;

    // Baseline distribution for novelty detection
    float* baseline_probs;        // [vocab_size] - running average
    float baseline_alpha;         // EMA decay for baseline

    // Async training (pthread)
    pthread_t training_thread;
    pthread_mutex_t delta_mutex;  // Protects delta during read/write
    int training_in_progress;     // 1 if async training running
    int total_training_cycles;    // Stats

    // Snapshot for async training thread (copied before launch)
    float* snapshot_mean_x;       // [dim] copy of mean_x at trigger time
    int* snapshot_target_counts;  // [vocab_size] copy of target_counts
    float snapshot_signal_avg;    // avg signal at trigger time
    int snapshot_experience_count;
} ExperienceAccumulator;

// Initialize accumulator
void init_accumulator(ExperienceAccumulator* acc, int dim, int vocab_size);

// Free accumulator
void free_accumulator(ExperienceAccumulator* acc);

// Accumulate one experience (instead of immediate training)
// Returns 1 if training was triggered, 0 otherwise
int accumulate_experience(ExperienceAccumulator* acc, MicroTrainer* mt,
                          LowRankDelta* delta, const float* x,
                          const float* probs, int target_id, float signal);

// Check if training should trigger and do it if ready
// Returns 1 if training was triggered
int maybe_trigger_training(ExperienceAccumulator* acc, MicroTrainer* mt,
                           LowRankDelta* delta);

// Force training with current buffer (used before shutdown)
void flush_accumulator(ExperienceAccumulator* acc, MicroTrainer* mt,
                       LowRankDelta* delta);

// Update cooldown (call each timestep with dt)
void accumulator_tick(ExperienceAccumulator* acc, float dt);

#endif // DELTA_H
