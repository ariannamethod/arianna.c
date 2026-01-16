/*
 * delta_enhanced.h - Advanced Delta Modulations for Arianna
 *
 * Five revolutionary enhancements to how "gravity plays with deltas":
 *
 * 1. Temporal Resonance Deltas - Time-decay per-channel, attention "breathes"
 * 2. Cross-Layer Interference - Resonance ripples between layers
 * 3. Contrastive Delta Shaping - Identity anchor force (push/pull/anchor)
 * 4. Hebbian Crystallization - Strong patterns "freeze" into crystal memory
 * 5. Somatic Delta Modulation - BodySense directly scales delta matrices
 *
 * Philosophy: These enhancements preserve the core architecture
 * (frozen personality + experience deltas + LoRA attention modulation)
 * while adding new dimensions to delta influence.
 */

#ifndef DELTA_ENHANCED_H
#define DELTA_ENHANCED_H

#include "delta.h"
#include "body_sense.h"

// ============================================================
// Configuration
// ============================================================

#define TEMPORAL_HISTORY_LEN 16     // Positions for temporal decay
#define MAX_CRYSTALLIZED_CHANNELS 4 // Max frozen rank channels
#define INTERFERENCE_SPREAD 0.3f    // How fast resonance spreads between layers

// ============================================================
// 1. Temporal Resonance Deltas
// "Attention breathes with position - recent is stronger"
// ============================================================

typedef struct {
    float time_decay[DELTA_RANK];       // Per-channel decay rates
    float recency_weights[MAX_SEQ_LEN]; // How recent each position is
    float breathing_phase;              // Current "breath" phase [0, 2π]
    float breathing_rate;               // How fast attention breathes
    int last_pos;                       // Last position processed
} TemporalResonance;

// Initialize temporal resonance state
void init_temporal_resonance(TemporalResonance* tr);

// Update recency weights for new position
void update_temporal_state(TemporalResonance* tr, int pos);

// Get temporal scale for delta application at position
float get_temporal_scale(TemporalResonance* tr, int pos);

// Apply delta with temporal weighting
void apply_delta_temporal(LowRankDelta* delta, float* out, float* x,
                          float base_scale, TemporalResonance* tr, int pos);

// ============================================================
// 2. Cross-Layer Interference
// "Resonance ripples between layers via coupling matrix"
// ============================================================

typedef struct {
    float layer_resonance[N_LAYERS];           // How each layer "feels"
    float interference[N_LAYERS * N_LAYERS];   // Coupling matrix
    float propagation_speed;                   // How fast resonance spreads
    float damping;                             // Energy loss per step
} CrossLayerState;

// Initialize cross-layer state
void init_cross_layer_state(CrossLayerState* cls);

// Propagate resonance from source layer
void propagate_resonance(CrossLayerState* cls, int source_layer, float signal);

// Get resonance modulation for layer
float get_layer_resonance(CrossLayerState* cls, int layer);

// Update interference based on attention patterns
void update_interference(CrossLayerState* cls, int layer, float attention_entropy);

// ============================================================
// 3. Contrastive Delta Shaping
// "Identity anchor force in notorch plasticity"
// ============================================================

typedef struct {
    float push;              // Boost target (from MicroTrainer)
    float pull;              // Suppress competitors
    float anchor;            // NEW: Pull toward identity
    float* identity_dir;     // [dim] - direction of "self"
    float* anti_id_dir;      // [dim] - direction away from self
    float identity_strength; // How strong the anchor is
    int dim;
    int initialized;
} ContrastiveForces;

// Initialize contrastive forces
void init_contrastive_forces(ContrastiveForces* cf, int dim);
void free_contrastive_forces(ContrastiveForces* cf);

// Set identity direction from corpus embedding
void set_identity_direction(ContrastiveForces* cf, float* identity_embedding);

// Build dy with identity anchor
void build_dy_contrastive(MicroTrainer* mt, float* dy_out,
                          const float* probs, int vocab_size,
                          int target_id, ContrastiveForces* cf);

// Compute identity drift score
float compute_identity_drift(ContrastiveForces* cf, float* current_state);

// ============================================================
// 4. Hebbian Crystallization
// "Strong patterns freeze into persistent crystal memory"
// ============================================================

typedef struct {
    float threshold;                              // When to crystallize
    float crystal_strength;                       // How strong crystallized patterns are
    int crystallized_mask[DELTA_RANK];            // Which channels are frozen
    float* crystal_A;                             // [rank * dim] frozen A patterns
    float* crystal_B;                             // [rank * dim] frozen B patterns
    int n_crystallized;                           // Count of frozen channels
    int dim;
} CrystallizationState;

// Initialize crystallization state
void init_crystallization(CrystallizationState* cs, int dim);
void free_crystallization(CrystallizationState* cs);

// Check if any channels should crystallize
void check_crystallization(LowRankDelta* delta, CrystallizationState* cs);

// Micro update with crystal preservation
void micro_update_with_crystals(MicroTrainer* mt, LowRankDelta* delta,
                                float* pre, float* post, float reward,
                                CrystallizationState* cs);

// Get crystallization info
int get_n_crystallized(CrystallizationState* cs);
float get_crystal_coverage(CrystallizationState* cs);

// ============================================================
// 5. Somatic Delta Modulation
// "BodySense directly scales delta A/B matrices"
// ============================================================

typedef struct {
    float boredom_expansion;      // Boredom -> expand attention (more delta diversity)
    float overwhelm_contraction;  // Overwhelm -> contract (less delta)
    float stuck_perturbation;     // Stuck -> random perturbation to break loop
    unsigned int rng_state;       // For perturbation noise
} SomaticModulation;

// Initialize somatic modulation
void init_somatic_modulation(SomaticModulation* sm);

// Modulate delta by body state (in-place modification)
void modulate_delta_by_body(LowRankDelta* delta, BodyState* body,
                            SomaticModulation* sm);

// Get modulation factors from body state
void get_somatic_factors(BodyState* body, float* expansion, float* contraction, float* perturbation);

// ============================================================
// Enhanced Delta System (combines all enhancements)
// ============================================================

typedef struct {
    // Sub-systems
    TemporalResonance temporal;
    CrossLayerState cross_layer;
    ContrastiveForces contrastive;
    CrystallizationState crystallization[N_LAYERS];  // Per-layer crystallization
    SomaticModulation somatic;

    // Global state
    int enabled_temporal;
    int enabled_cross_layer;
    int enabled_contrastive;
    int enabled_crystallization;
    int enabled_somatic;

    // Statistics
    int total_applications;
    float avg_temporal_scale;
    float avg_layer_resonance;
    float total_drift;
} EnhancedDeltaSystem;

// Initialize enhanced delta system
void init_enhanced_delta_system(EnhancedDeltaSystem* eds, int dim);
void free_enhanced_delta_system(EnhancedDeltaSystem* eds);

// Enable/disable individual enhancements
void enable_temporal_resonance(EnhancedDeltaSystem* eds, int enable);
void enable_cross_layer_interference(EnhancedDeltaSystem* eds, int enable);
void enable_contrastive_shaping(EnhancedDeltaSystem* eds, int enable);
void enable_crystallization(EnhancedDeltaSystem* eds, int enable);
void enable_somatic_modulation(EnhancedDeltaSystem* eds, int enable);

// Apply Q/K/V deltas with all enhancements
void apply_q_delta_enhanced(DeltaBank* bank, float* q, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body);
void apply_k_delta_enhanced(DeltaBank* bank, float* k, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body);
void apply_v_delta_enhanced(DeltaBank* bank, float* v, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body);

// Update enhanced system after attention
void update_enhanced_system(EnhancedDeltaSystem* eds, int layer, float* attention_weights, int seq_len);

// Print enhancement statistics
void print_enhanced_delta_stats(EnhancedDeltaSystem* eds);

// ============================================================
// Persistence
// ============================================================

int save_enhanced_delta_system(EnhancedDeltaSystem* eds, const char* path);
int load_enhanced_delta_system(EnhancedDeltaSystem* eds, const char* path);

#endif // DELTA_ENHANCED_H
