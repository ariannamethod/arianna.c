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
    float* A;       // [out_dim, rank]
    float* B;       // [rank, in_dim]
    int out_dim;
    int in_dim;
    int rank;
} LowRankDelta;

// Experience shard - collection of deltas for all layers
typedef struct {
    char name[64];

    // Attention deltas per layer: modify Q,K,V projections
    LowRankDelta* attn_q_deltas;  // [n_layers]
    LowRankDelta* attn_k_deltas;  // [n_layers]
    LowRankDelta* attn_v_deltas;  // [n_layers]

    // Metadata
    float strength;              // How much this shard influences
    int n_layers;
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

#endif // DELTA_H
