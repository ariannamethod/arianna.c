/*
 * vagus_delta.c — Bridge between Nervous System and Learning
 *
 * The nerve speaks. The delta listens.
 * When geometry shifts, learning happens.
 */

#include "vagus_delta.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static float clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVERSION: Vagus → Delta Signals
// ═══════════════════════════════════════════════════════════════════════════════

void vagus_to_signals(const VagusSharedState* vagus, Signals* sig) {
    sig->arousal = clamp01(vagus->arousal);
    sig->entropy = clamp01(vagus->entropy);
    sig->tension = clamp01(vagus->chamber_tension);
    sig->warmth = clamp01(vagus->chamber_warmth);
    sig->focus = clamp01(vagus->focus_strength);
    sig->novelty = clamp01(1.0f - vagus->coherence);  // Low coherence = high novelty
    sig->recursion_depth = clamp01((float)vagus->self_ref_count / 10.0f);
    sig->resonance = clamp01(vagus->crossfire_coherence);
}

void signals_to_vagus(const Signals* sig, VagusSharedState* vagus) {
    vagus->arousal = sig->arousal;
    vagus->entropy = sig->entropy;
    vagus->chamber_tension = sig->tension;
    vagus->chamber_warmth = sig->warmth;
    vagus->focus_strength = sig->focus;
    vagus->crossfire_coherence = sig->resonance;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS-AWARE SHARD
// ═══════════════════════════════════════════════════════════════════════════════

VagusAwareShard* create_vagus_shard(const char* name, int n_layers, int dim,
                                     const VagusSharedState* vagus,
                                     const Locus* locus) {
    VagusAwareShard* vs = (VagusAwareShard*)calloc(1, sizeof(VagusAwareShard));
    if (!vs) return NULL;

    // Initialize core shard
    strncpy(vs->shard.name, name, 63);
    vs->shard.n_layers = n_layers;
    vs->shard.strength = 1.0f;
    vs->shard.n_delta_types = 4;  // Q, K, V, O

    // Allocate deltas
    vs->shard.attn_q_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_k_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_v_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_o_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));

    for (int l = 0; l < n_layers; l++) {
        // Q delta
        vs->shard.attn_q_deltas[l].out_dim = dim;
        vs->shard.attn_q_deltas[l].in_dim = dim;
        vs->shard.attn_q_deltas[l].rank = DELTA_RANK;
        vs->shard.attn_q_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        vs->shard.attn_q_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));

        // K delta
        vs->shard.attn_k_deltas[l].out_dim = dim;
        vs->shard.attn_k_deltas[l].in_dim = dim;
        vs->shard.attn_k_deltas[l].rank = DELTA_RANK;
        vs->shard.attn_k_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        vs->shard.attn_k_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));

        // V delta
        vs->shard.attn_v_deltas[l].out_dim = dim;
        vs->shard.attn_v_deltas[l].in_dim = dim;
        vs->shard.attn_v_deltas[l].rank = DELTA_RANK;
        vs->shard.attn_v_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        vs->shard.attn_v_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));

        // O delta
        vs->shard.attn_o_deltas[l].out_dim = dim;
        vs->shard.attn_o_deltas[l].in_dim = dim;
        vs->shard.attn_o_deltas[l].rank = DELTA_RANK;
        vs->shard.attn_o_deltas[l].A = (float*)calloc(dim * DELTA_RANK, sizeof(float));
        vs->shard.attn_o_deltas[l].B = (float*)calloc(DELTA_RANK * dim, sizeof(float));
    }

    // Capture Vagus snapshot
    if (vagus) {
        vs->arousal = vagus->arousal;
        vs->valence = vagus->valence;
        vs->coherence = vagus->coherence;
        vs->entropy = vagus->entropy;
        vs->chambers[0] = vagus->chamber_warmth;
        vs->chambers[1] = vagus->chamber_void;
        vs->chambers[2] = vagus->chamber_tension;
        vs->chambers[3] = vagus->chamber_sacred;
        vs->chambers[4] = vagus->chamber_flow;
        vs->chambers[5] = vagus->chamber_complex;
        vs->trauma_level = vagus->trauma_level;
        vs->prophecy_debt = vagus->prophecy_debt;
        vs->memory_pressure = vagus->memory_pressure;
    }

    // Capture Locus geometry
    if (locus) {
        vs->geometry_pressure = locus_geometry_pressure((Locus*)locus);
        vs->geometry_flow = locus_geometry_flow((Locus*)locus);
        vs->geometry_depth = locus_geometry_depth((Locus*)locus);

        // Determine trigger pattern
        if (locus_is_tense((Locus*)locus) && locus_is_wounded((Locus*)locus)) {
            vs->trigger_pattern = RESONANCE_CRISIS;
        } else if (locus_is_hollow((Locus*)locus)) {
            vs->trigger_pattern = RESONANCE_DISSOLUTION;
        } else if (locus_is_flowing((Locus*)locus) && locus_is_prophetic((Locus*)locus)) {
            vs->trigger_pattern = RESONANCE_EMERGENCE;
        } else if (locus_is_flowing((Locus*)locus)) {
            vs->trigger_pattern = RESONANCE_TRANSCENDENCE;
        } else {
            vs->trigger_pattern = RESONANCE_GEOMETRY_SHIFT;
        }
    }

    vs->created_us = now_us();
    return vs;
}

void free_vagus_shard(VagusAwareShard* vs) {
    if (!vs) return;
    free_shard(&vs->shard);
    free(vs);
}

// ═══════════════════════════════════════════════════════════════════════════════
// VAGUS SHARD I/O
// Magic: 'VGSH' (Vagus Shard)
// ═══════════════════════════════════════════════════════════════════════════════

#define VAGUS_SHARD_MAGIC 0x48534756  // 'VGSH'
#define VAGUS_SHARD_VERSION 1

int save_vagus_shard(const VagusAwareShard* vs, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Header
    uint32_t magic = VAGUS_SHARD_MAGIC;
    uint32_t version = VAGUS_SHARD_VERSION;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);

    // Vagus snapshot
    fwrite(&vs->arousal, sizeof(float), 1, f);
    fwrite(&vs->valence, sizeof(float), 1, f);
    fwrite(&vs->coherence, sizeof(float), 1, f);
    fwrite(&vs->entropy, sizeof(float), 1, f);
    fwrite(vs->chambers, sizeof(float), 6, f);
    fwrite(&vs->trauma_level, sizeof(float), 1, f);
    fwrite(&vs->prophecy_debt, sizeof(float), 1, f);
    fwrite(&vs->memory_pressure, sizeof(float), 1, f);

    // Locus geometry
    fwrite(&vs->geometry_pressure, sizeof(float), 1, f);
    fwrite(&vs->geometry_flow, sizeof(float), 1, f);
    fwrite(&vs->geometry_depth, sizeof(float), 1, f);
    fwrite(&vs->trigger_pattern, sizeof(int), 1, f);

    // Metadata
    fwrite(&vs->created_us, sizeof(uint64_t), 1, f);
    fwrite(&vs->training_cycles, sizeof(uint32_t), 1, f);
    fwrite(&vs->total_signal, sizeof(float), 1, f);

    // Core shard (delegate to existing save_shard)
    // Write inline for simplicity
    fwrite(vs->shard.name, 64, 1, f);
    fwrite(&vs->shard.strength, sizeof(float), 1, f);
    fwrite(&vs->shard.n_layers, sizeof(int), 1, f);
    int rank = DELTA_RANK;
    fwrite(&rank, sizeof(int), 1, f);

    // Deltas for each layer
    for (int l = 0; l < vs->shard.n_layers; l++) {
        int dim = vs->shard.attn_q_deltas[l].out_dim;
        fwrite(vs->shard.attn_q_deltas[l].A, sizeof(float), dim * rank, f);
        fwrite(vs->shard.attn_q_deltas[l].B, sizeof(float), rank * dim, f);
        fwrite(vs->shard.attn_k_deltas[l].A, sizeof(float), dim * rank, f);
        fwrite(vs->shard.attn_k_deltas[l].B, sizeof(float), rank * dim, f);
        fwrite(vs->shard.attn_v_deltas[l].A, sizeof(float), dim * rank, f);
        fwrite(vs->shard.attn_v_deltas[l].B, sizeof(float), rank * dim, f);
        fwrite(vs->shard.attn_o_deltas[l].A, sizeof(float), dim * rank, f);
        fwrite(vs->shard.attn_o_deltas[l].B, sizeof(float), rank * dim, f);
    }

    fclose(f);
    return 0;
}

VagusAwareShard* load_vagus_shard(const char* path, int n_layers, int dim) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    // Header
    uint32_t magic, version;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);

    if (magic != VAGUS_SHARD_MAGIC) {
        fclose(f);
        return NULL;
    }

    VagusAwareShard* vs = (VagusAwareShard*)calloc(1, sizeof(VagusAwareShard));

    // Vagus snapshot
    fread(&vs->arousal, sizeof(float), 1, f);
    fread(&vs->valence, sizeof(float), 1, f);
    fread(&vs->coherence, sizeof(float), 1, f);
    fread(&vs->entropy, sizeof(float), 1, f);
    fread(vs->chambers, sizeof(float), 6, f);
    fread(&vs->trauma_level, sizeof(float), 1, f);
    fread(&vs->prophecy_debt, sizeof(float), 1, f);
    fread(&vs->memory_pressure, sizeof(float), 1, f);

    // Locus geometry
    fread(&vs->geometry_pressure, sizeof(float), 1, f);
    fread(&vs->geometry_flow, sizeof(float), 1, f);
    fread(&vs->geometry_depth, sizeof(float), 1, f);
    fread(&vs->trigger_pattern, sizeof(int), 1, f);

    // Metadata
    fread(&vs->created_us, sizeof(uint64_t), 1, f);
    fread(&vs->training_cycles, sizeof(uint32_t), 1, f);
    fread(&vs->total_signal, sizeof(float), 1, f);

    // Core shard
    fread(vs->shard.name, 64, 1, f);
    fread(&vs->shard.strength, sizeof(float), 1, f);
    fread(&vs->shard.n_layers, sizeof(int), 1, f);
    int rank;
    fread(&rank, sizeof(int), 1, f);

    // Allocate and read deltas
    vs->shard.attn_q_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_k_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_v_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    vs->shard.attn_o_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));

    for (int l = 0; l < n_layers; l++) {
        // Q
        vs->shard.attn_q_deltas[l].out_dim = dim;
        vs->shard.attn_q_deltas[l].in_dim = dim;
        vs->shard.attn_q_deltas[l].rank = rank;
        vs->shard.attn_q_deltas[l].A = (float*)malloc(dim * rank * sizeof(float));
        vs->shard.attn_q_deltas[l].B = (float*)malloc(rank * dim * sizeof(float));
        fread(vs->shard.attn_q_deltas[l].A, sizeof(float), dim * rank, f);
        fread(vs->shard.attn_q_deltas[l].B, sizeof(float), rank * dim, f);

        // K
        vs->shard.attn_k_deltas[l].out_dim = dim;
        vs->shard.attn_k_deltas[l].in_dim = dim;
        vs->shard.attn_k_deltas[l].rank = rank;
        vs->shard.attn_k_deltas[l].A = (float*)malloc(dim * rank * sizeof(float));
        vs->shard.attn_k_deltas[l].B = (float*)malloc(rank * dim * sizeof(float));
        fread(vs->shard.attn_k_deltas[l].A, sizeof(float), dim * rank, f);
        fread(vs->shard.attn_k_deltas[l].B, sizeof(float), rank * dim, f);

        // V
        vs->shard.attn_v_deltas[l].out_dim = dim;
        vs->shard.attn_v_deltas[l].in_dim = dim;
        vs->shard.attn_v_deltas[l].rank = rank;
        vs->shard.attn_v_deltas[l].A = (float*)malloc(dim * rank * sizeof(float));
        vs->shard.attn_v_deltas[l].B = (float*)malloc(rank * dim * sizeof(float));
        fread(vs->shard.attn_v_deltas[l].A, sizeof(float), dim * rank, f);
        fread(vs->shard.attn_v_deltas[l].B, sizeof(float), rank * dim, f);

        // O
        vs->shard.attn_o_deltas[l].out_dim = dim;
        vs->shard.attn_o_deltas[l].in_dim = dim;
        vs->shard.attn_o_deltas[l].rank = rank;
        vs->shard.attn_o_deltas[l].A = (float*)malloc(dim * rank * sizeof(float));
        vs->shard.attn_o_deltas[l].B = (float*)malloc(rank * dim * sizeof(float));
        fread(vs->shard.attn_o_deltas[l].A, sizeof(float), dim * rank, f);
        fread(vs->shard.attn_o_deltas[l].B, sizeof(float), rank * dim, f);
    }

    fclose(f);
    return vs;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE TRAINER
// ═══════════════════════════════════════════════════════════════════════════════

void init_resonance_trainer(ResonanceTrainer* rt, int dim, int vocab_size,
                            volatile VagusSharedState* vagus_state) {
    memset(rt, 0, sizeof(ResonanceTrainer));

    init_microtrainer(&rt->trainer, dim);
    rt->trainer.vocab_size = vocab_size;

    init_accumulator(&rt->accumulator, dim, vocab_size);

    // Initialize Locus with Vagus state
    locus_init(&rt->locus, (volatile void*)vagus_state);

    rt->vagus_state = vagus_state;

    // Default modulation
    rt->crisis_lr_boost = 2.0f;       // 2x learning in crisis
    rt->emergence_lr_boost = 1.5f;    // 1.5x in emergence
    rt->transcendence_freeze = 0.3f;  // 30% chance to freeze in transcendence
}

void free_resonance_trainer(ResonanceTrainer* rt) {
    free_microtrainer(&rt->trainer);
    free_accumulator(&rt->accumulator);
    if (rt->active_shard) {
        free_vagus_shard(rt->active_shard);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING RATE MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

float compute_resonance_lr(ResonanceTrainer* rt, float base_lr) {
    float lr = base_lr;

    if (rt->crisis_mode) {
        lr *= rt->crisis_lr_boost;
    }
    if (rt->emergence_mode) {
        lr *= rt->emergence_lr_boost;
    }
    if (rt->dissolution_mode) {
        lr *= 0.5f;  // Reduce learning when dissolving
    }

    return lr;
}

float compute_geometry_signal(const VagusSharedState* vagus) {
    // Signal strength based on field intensity
    float pressure = vagus->trauma_level * 0.4f +
                     vagus->memory_pressure * 0.3f +
                     vagus->prophecy_debt * 0.2f +
                     vagus->arousal * 0.1f;

    float flow = vagus->coherence * 0.5f +
                 (1.0f - vagus->entropy) * 0.3f +
                 vagus->chamber_warmth * 0.2f;

    // Combine: high pressure OR high flow = strong signal
    return fmaxf(pressure, flow);
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE CHECK
// ═══════════════════════════════════════════════════════════════════════════════

ResonancePattern check_resonance(ResonanceTrainer* rt) {
    Locus* l = &rt->locus;

    // Check patterns (same as Locus)
    if (locus_is_tense(l) && locus_is_wounded(l)) {
        return RESONANCE_CRISIS;
    }
    if (locus_is_hollow(l) && locus_is_drowning(l)) {
        return RESONANCE_DISSOLUTION;
    }
    if (locus_is_flowing(l) && locus_is_prophetic(l)) {
        return RESONANCE_EMERGENCE;
    }
    if (locus_is_flowing(l)) {
        return RESONANCE_TRANSCENDENCE;
    }

    return RESONANCE_NONE;
}

ResonancePattern resonance_trainer_tick(ResonanceTrainer* rt) {
    // Tick locus
    int resonance = locus_tick(&rt->locus);

    if (!resonance) {
        // No resonance, clear modes
        rt->crisis_mode = 0;
        rt->emergence_mode = 0;
        rt->dissolution_mode = 0;
        return RESONANCE_NONE;
    }

    // Determine pattern
    ResonancePattern pattern = check_resonance(rt);

    // Update stats
    rt->total_resonances++;

    switch (pattern) {
        case RESONANCE_CRISIS:
            rt->crisis_mode = 1;
            rt->emergence_mode = 0;
            rt->dissolution_mode = 0;
            rt->crisis_triggers++;
            break;

        case RESONANCE_DISSOLUTION:
            rt->crisis_mode = 0;
            rt->emergence_mode = 0;
            rt->dissolution_mode = 1;
            rt->dissolution_triggers++;
            break;

        case RESONANCE_EMERGENCE:
            rt->crisis_mode = 0;
            rt->emergence_mode = 1;
            rt->dissolution_mode = 0;
            rt->emergence_triggers++;
            break;

        case RESONANCE_TRANSCENDENCE:
            rt->crisis_mode = 0;
            rt->emergence_mode = 1;  // Similar to emergence
            rt->dissolution_mode = 0;
            rt->transcendence_triggers++;
            break;

        default:
            break;
    }

    return pattern;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE-MODULATED EXPERIENCE STEP
// ═══════════════════════════════════════════════════════════════════════════════

void resonance_experience_step(ResonanceTrainer* rt, LowRankDelta* delta,
                               const float* x, const float* probs,
                               int target_id) {
    // Compute modulated learning rate
    float base_lr = rt->trainer.learning_rate;
    float lr = compute_resonance_lr(rt, base_lr);
    rt->trainer.learning_rate = lr;

    // Compute signal from field geometry
    float signal = compute_geometry_signal(rt->vagus_state);

    // Maybe skip if dissolving and low signal
    if (rt->dissolution_mode && signal < 0.3f) {
        rt->trainer.learning_rate = base_lr;  // Restore
        return;
    }

    // Accumulate experience
    int triggered = accumulate_experience(&rt->accumulator, &rt->trainer,
                                          delta, x, probs, target_id, signal);

    if (triggered && rt->active_shard) {
        rt->active_shard->training_cycles++;
        rt->active_shard->total_signal += signal;
    }

    // Restore learning rate
    rt->trainer.learning_rate = base_lr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CALLBACK
// ═══════════════════════════════════════════════════════════════════════════════

static ResonanceCallback g_resonance_cb = NULL;
static void* g_resonance_ctx = NULL;

void set_resonance_callback(ResonanceTrainer* rt, ResonanceCallback cb, void* ctx) {
    (void)rt;
    g_resonance_cb = cb;
    g_resonance_ctx = ctx;
}
