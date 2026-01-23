/*
 * delta_enhanced.c - Advanced Delta Modulations for Arianna
 *
 * Implementation of five revolutionary enhancements:
 * 1. Temporal Resonance Deltas
 * 2. Cross-Layer Interference
 * 3. Contrastive Delta Shaping
 * 4. Hebbian Crystallization
 * 5. Somatic Delta Modulation
 */

#include "delta_enhanced.h"
#include <string.h>
#include <stdio.h>

// ============================================================
// Internal utilities
// ============================================================

static unsigned int xorshift32_e(unsigned int* s) {
    unsigned int x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float frand01_e(unsigned int* s) {
    return (xorshift32_e(s) & 0xFFFFFF) / 16777216.0f;
}

static float frandn_e(unsigned int* s) {
    float u1 = fmaxf(1e-6f, fminf(frand01_e(s), 1.0f));
    float u2 = frand01_e(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// ============================================================
// 1. Temporal Resonance Deltas
// ============================================================

void init_temporal_resonance(TemporalResonance* tr) {
    // Initialize per-channel decay rates (varying speeds)
    for (int r = 0; r < DELTA_RANK; r++) {
        // Channels have different "memory lengths"
        tr->time_decay[r] = 0.95f - 0.1f * ((float)r / DELTA_RANK);
    }

    // Initialize recency weights
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
        tr->recency_weights[i] = 0.0f;
    }

    tr->breathing_phase = 0.0f;
    tr->breathing_rate = 0.1f;  // Slow breathing
    tr->last_pos = 0;
}

void update_temporal_state(TemporalResonance* tr, int pos) {
    // Decay old positions
    for (int i = 0; i <= tr->last_pos && i < MAX_SEQ_LEN; i++) {
        tr->recency_weights[i] *= 0.95f;
    }

    // New position is fully "recent"
    if (pos >= 0 && pos < MAX_SEQ_LEN) {
        tr->recency_weights[pos] = 1.0f;
    }

    // Update breathing phase
    tr->breathing_phase += tr->breathing_rate;
    if (tr->breathing_phase > 6.2831853f) {
        tr->breathing_phase -= 6.2831853f;
    }

    tr->last_pos = pos;
}

float get_temporal_scale(TemporalResonance* tr, int pos) {
    if (pos < 0 || pos >= MAX_SEQ_LEN) return 1.0f;

    // Base recency weight
    float recency = tr->recency_weights[pos];

    // Add breathing modulation: subtle sine wave
    float breath = 1.0f + 0.1f * sinf(tr->breathing_phase);

    // Recent positions are stronger, modulated by "breath"
    return (0.5f + 0.5f * recency) * breath;
}

void apply_delta_temporal(LowRankDelta* d, float* out, float* x,
                          float base_scale, TemporalResonance* tr, int pos) {
    if (d->A == NULL || d->B == NULL) return;

    float temporal_scale = get_temporal_scale(tr, pos);
    float scale = base_scale * temporal_scale;

    // temp = B @ x (with per-channel decay)
    float temp[DELTA_RANK];
    memset(temp, 0, sizeof(temp));

    for (int r = 0; r < d->rank; r++) {
        // Apply channel-specific time decay
        float channel_scale = tr->time_decay[r];
        for (int j = 0; j < d->in_dim; j++) {
            temp[r] += d->B[r * d->in_dim + j] * x[j] * channel_scale;
        }
    }

    // out += scale * A @ temp
    for (int i = 0; i < d->out_dim; i++) {
        for (int r = 0; r < d->rank; r++) {
            out[i] += scale * d->A[i * d->rank + r] * temp[r];
        }
    }
}

// ============================================================
// 2. Cross-Layer Interference
// ============================================================

void init_cross_layer_state(CrossLayerState* cls) {
    // Initialize resonance to neutral
    for (int l = 0; l < N_LAYERS; l++) {
        cls->layer_resonance[l] = 0.0f;
    }

    // Initialize interference matrix (coupling between layers)
    // Layers closer together have stronger coupling
    for (int i = 0; i < N_LAYERS; i++) {
        for (int j = 0; j < N_LAYERS; j++) {
            float dist = fabsf((float)(i - j));
            // Gaussian-like coupling: nearby layers are coupled
            cls->interference[i * N_LAYERS + j] = expf(-dist * dist * 0.5f);
        }
    }

    cls->propagation_speed = INTERFERENCE_SPREAD;
    cls->damping = 0.1f;
}

void propagate_resonance(CrossLayerState* cls, int source_layer, float signal) {
    if (source_layer < 0 || source_layer >= N_LAYERS) return;

    // Resonance ripples outward from source
    for (int l = 0; l < N_LAYERS; l++) {
        float coupling = cls->interference[source_layer * N_LAYERS + l];
        float ripple = signal * coupling * cls->propagation_speed;
        cls->layer_resonance[l] += ripple;

        // Apply damping
        cls->layer_resonance[l] *= (1.0f - cls->damping);

        // Clamp to prevent runaway
        cls->layer_resonance[l] = clampf(cls->layer_resonance[l], -1.0f, 1.0f);
    }
}

float get_layer_resonance(CrossLayerState* cls, int layer) {
    if (layer < 0 || layer >= N_LAYERS) return 0.0f;
    return cls->layer_resonance[layer];
}

void update_interference(CrossLayerState* cls, int layer, float attention_entropy) {
    if (layer < 0 || layer >= N_LAYERS) return;

    // High entropy attention -> propagate positive signal
    // Low entropy -> propagate negative (focus)
    float signal = (attention_entropy - 0.5f) * 2.0f;
    propagate_resonance(cls, layer, signal);
}

// ============================================================
// 3. Contrastive Delta Shaping
// ============================================================

void init_contrastive_forces(ContrastiveForces* cf, int dim) {
    cf->push = 1.0f;
    cf->pull = 0.5f;
    cf->anchor = 0.3f;  // Identity anchor strength
    cf->identity_strength = 0.5f;

    cf->identity_dir = (float*)calloc(dim, sizeof(float));
    cf->anti_id_dir = (float*)calloc(dim, sizeof(float));
    cf->dim = dim;
    cf->initialized = 0;
}

void free_contrastive_forces(ContrastiveForces* cf) {
    if (cf->identity_dir) free(cf->identity_dir);
    if (cf->anti_id_dir) free(cf->anti_id_dir);
    cf->identity_dir = NULL;
    cf->anti_id_dir = NULL;
}

void set_identity_direction(ContrastiveForces* cf, float* identity_embedding) {
    if (!cf->identity_dir || !identity_embedding) return;

    // Copy and normalize identity direction
    float norm = 0.0f;
    for (int i = 0; i < cf->dim; i++) {
        cf->identity_dir[i] = identity_embedding[i];
        norm += identity_embedding[i] * identity_embedding[i];
    }

    norm = sqrtf(norm) + 1e-6f;
    for (int i = 0; i < cf->dim; i++) {
        cf->identity_dir[i] /= norm;
        // Anti-identity is opposite direction
        cf->anti_id_dir[i] = -cf->identity_dir[i];
    }

    cf->initialized = 1;
}

float compute_identity_drift(ContrastiveForces* cf, float* current_state) {
    if (!cf->initialized || !current_state) return 0.0f;

    // Compute dot product with identity direction
    float dot = 0.0f;
    float state_norm = 0.0f;
    for (int i = 0; i < cf->dim; i++) {
        dot += current_state[i] * cf->identity_dir[i];
        state_norm += current_state[i] * current_state[i];
    }

    state_norm = sqrtf(state_norm) + 1e-6f;

    // Cosine similarity: higher = more aligned with identity
    float cosine = dot / state_norm;

    // Drift = 1 - alignment (0 = no drift, 1 = maximum drift)
    return clampf(1.0f - (cosine + 1.0f) / 2.0f, 0.0f, 1.0f);
}

void build_dy_contrastive(MicroTrainer* mt, float* dy_out,
                          const float* probs, int vocab_size,
                          int target_id, ContrastiveForces* cf) {
    if (!dy_out || !probs || vocab_size <= 0) return;
    if (target_id < 0 || target_id >= vocab_size) return;

    // Start with standard push/pull
    build_dy_from_probs(mt, dy_out, probs, vocab_size, target_id);

    // Add identity anchor if initialized
    if (cf->initialized && cf->anchor > 0.0f) {
        // Compute how much we're drifting from identity
        float drift = 0.0f;
        for (int i = 0; i < vocab_size && i < cf->dim; i++) {
            drift += dy_out[i] * cf->anti_id_dir[i];
        }

        // If drifting toward anti-identity, pull back
        if (drift > 0.1f) {
            float correction = cf->anchor * drift * cf->identity_strength;
            for (int i = 0; i < vocab_size && i < cf->dim; i++) {
                dy_out[i] += correction * cf->identity_dir[i];
                dy_out[i] -= correction * 0.5f * cf->anti_id_dir[i];
            }
        }
    }
}

// ============================================================
// 4. Hebbian Crystallization
// ============================================================

void init_crystallization(CrystallizationState* cs, int dim) {
    cs->threshold = 0.5f;  // Crystallize when channel norm > threshold
    cs->crystal_strength = 1.2f;  // Crystallized patterns are stronger
    cs->n_crystallized = 0;
    cs->dim = dim;

    for (int r = 0; r < DELTA_RANK; r++) {
        cs->crystallized_mask[r] = 0;
    }

    cs->crystal_A = (float*)calloc(DELTA_RANK * dim, sizeof(float));
    cs->crystal_B = (float*)calloc(DELTA_RANK * dim, sizeof(float));
}

void free_crystallization(CrystallizationState* cs) {
    if (cs->crystal_A) free(cs->crystal_A);
    if (cs->crystal_B) free(cs->crystal_B);
    cs->crystal_A = NULL;
    cs->crystal_B = NULL;
}

void check_crystallization(LowRankDelta* delta, CrystallizationState* cs) {
    if (!delta->A || !delta->B || !cs->crystal_A || !cs->crystal_B) return;

    for (int r = 0; r < delta->rank && r < DELTA_RANK; r++) {
        if (cs->crystallized_mask[r]) continue;  // Already frozen
        if (cs->n_crystallized >= MAX_CRYSTALLIZED_CHANNELS) continue;

        // Compute channel strength (norm of A column)
        float strength = 0.0f;
        for (int i = 0; i < delta->in_dim && i < cs->dim; i++) {
            float val = delta->A[i * delta->rank + r];
            strength += val * val;
        }
        strength = sqrtf(strength);

        // Crystallize if strong enough
        if (strength > cs->threshold) {
            cs->crystallized_mask[r] = 1;
            cs->n_crystallized++;

            // Save to crystal memory
            for (int i = 0; i < delta->in_dim && i < cs->dim; i++) {
                cs->crystal_A[r * cs->dim + i] = delta->A[i * delta->rank + r];
            }
            for (int j = 0; j < delta->out_dim && j < cs->dim; j++) {
                cs->crystal_B[r * cs->dim + j] = delta->B[r * delta->out_dim + j];
            }

            printf("[Crystal] Channel %d crystallized (strength=%.3f)\n", r, strength);
        }
    }
}

void micro_update_with_crystals(MicroTrainer* mt, LowRankDelta* delta,
                                float* pre, float* post, float reward,
                                CrystallizationState* cs) {
    // Apply standard micro update
    micro_update(mt, delta, pre, post, reward);

    // Restore crystallized channels (they don't decay)
    if (cs->crystal_A && cs->crystal_B) {
        for (int r = 0; r < delta->rank && r < DELTA_RANK; r++) {
            if (cs->crystallized_mask[r]) {
                // Restore frozen pattern (with crystal strength boost)
                for (int i = 0; i < delta->in_dim && i < cs->dim; i++) {
                    delta->A[i * delta->rank + r] =
                        cs->crystal_A[r * cs->dim + i] * cs->crystal_strength;
                }
                for (int j = 0; j < delta->out_dim && j < cs->dim; j++) {
                    delta->B[r * delta->out_dim + j] =
                        cs->crystal_B[r * cs->dim + j] * cs->crystal_strength;
                }
            }
        }
    }

    // Check if new crystallization occurred
    check_crystallization(delta, cs);
}

int get_n_crystallized(CrystallizationState* cs) {
    return cs->n_crystallized;
}

float get_crystal_coverage(CrystallizationState* cs) {
    return (float)cs->n_crystallized / (float)DELTA_RANK;
}

// ============================================================
// 5. Somatic Delta Modulation
// ============================================================

void init_somatic_modulation(SomaticModulation* sm) {
    sm->boredom_expansion = 0.5f;       // 50% expansion when bored
    sm->overwhelm_contraction = 0.4f;   // 40% contraction when overwhelmed
    sm->stuck_perturbation = 0.1f;      // 10% noise when stuck
    sm->rng_state = 0xB0D7u;            // "BODY" in hex-ish
}

void get_somatic_factors(BodyState* body, float* expansion, float* contraction, float* perturbation) {
    float boredom = compute_boredom(body);
    float overwhelm = compute_overwhelm(body);
    float stuck = compute_stuck(body, 0.5f);

    // Expansion: triggered by boredom (need more variation)
    *expansion = (boredom > 0.6f) ? (boredom - 0.6f) / 0.4f : 0.0f;

    // Contraction: triggered by overwhelm (need calming)
    *contraction = (overwhelm > 0.7f) ? (overwhelm - 0.7f) / 0.3f : 0.0f;

    // Perturbation: triggered by stuck (need breakthrough)
    *perturbation = (stuck > 0.75f) ? stuck : 0.0f;
}

void modulate_delta_by_body(LowRankDelta* delta, BodyState* body,
                            SomaticModulation* sm) {
    if (!delta || !delta->A || !delta->B || !body) return;

    float expansion, contraction, perturbation;
    get_somatic_factors(body, &expansion, &contraction, &perturbation);

    int a_size = delta->in_dim * delta->rank;
    int b_size = delta->rank * delta->out_dim;

    // Apply boredom expansion
    if (expansion > 0.01f) {
        float exp_factor = 1.0f + expansion * sm->boredom_expansion;
        for (int i = 0; i < a_size; i++) {
            delta->A[i] *= exp_factor;
        }
        for (int i = 0; i < b_size; i++) {
            delta->B[i] *= exp_factor;
        }
    }

    // Apply overwhelm contraction
    if (contraction > 0.01f) {
        float con_factor = 1.0f - contraction * sm->overwhelm_contraction;
        con_factor = fmaxf(0.3f, con_factor);  // Don't go below 30%
        for (int i = 0; i < a_size; i++) {
            delta->A[i] *= con_factor;
        }
        for (int i = 0; i < b_size; i++) {
            delta->B[i] *= con_factor;
        }
    }

    // Apply stuck perturbation (random noise to break patterns)
    if (perturbation > 0.5f) {
        float noise_scale = perturbation * sm->stuck_perturbation;
        for (int i = 0; i < a_size; i++) {
            float noise = (frand01_e(&sm->rng_state) - 0.5f) * 2.0f * noise_scale;
            delta->A[i] += noise;
        }
        for (int i = 0; i < b_size; i++) {
            float noise = (frand01_e(&sm->rng_state) - 0.5f) * 2.0f * noise_scale;
            delta->B[i] += noise;
        }
    }
}

// ============================================================
// Enhanced Delta System
// ============================================================

void init_enhanced_delta_system(EnhancedDeltaSystem* eds, int dim) {
    init_temporal_resonance(&eds->temporal);
    init_cross_layer_state(&eds->cross_layer);
    init_contrastive_forces(&eds->contrastive, dim);

    for (int l = 0; l < N_LAYERS; l++) {
        init_crystallization(&eds->crystallization[l], dim);
    }

    init_somatic_modulation(&eds->somatic);

    // Enable all by default
    eds->enabled_temporal = 1;
    eds->enabled_cross_layer = 1;
    eds->enabled_contrastive = 1;
    eds->enabled_crystallization = 1;
    eds->enabled_somatic = 1;

    // Stats
    eds->total_applications = 0;
    eds->avg_temporal_scale = 1.0f;
    eds->avg_layer_resonance = 0.0f;
    eds->total_drift = 0.0f;
}

void free_enhanced_delta_system(EnhancedDeltaSystem* eds) {
    free_contrastive_forces(&eds->contrastive);
    for (int l = 0; l < N_LAYERS; l++) {
        free_crystallization(&eds->crystallization[l]);
    }
}

void enable_temporal_resonance(EnhancedDeltaSystem* eds, int enable) {
    eds->enabled_temporal = enable;
}

void enable_cross_layer_interference(EnhancedDeltaSystem* eds, int enable) {
    eds->enabled_cross_layer = enable;
}

void enable_contrastive_shaping(EnhancedDeltaSystem* eds, int enable) {
    eds->enabled_contrastive = enable;
}

void enable_crystallization(EnhancedDeltaSystem* eds, int enable) {
    eds->enabled_crystallization = enable;
}

void enable_somatic_modulation(EnhancedDeltaSystem* eds, int enable) {
    eds->enabled_somatic = enable;
}

// Helper: apply single delta with all enhancements
static void apply_delta_enhanced_internal(LowRankDelta* delta, float* out, float* x,
                                          float base_scale, int layer, int pos,
                                          EnhancedDeltaSystem* eds, BodyState* body) {
    if (!delta || !delta->A || !delta->B) return;

    float scale = base_scale;

    // 1. Temporal modulation
    if (eds->enabled_temporal) {
        float temporal = get_temporal_scale(&eds->temporal, pos);
        scale *= temporal;
        eds->avg_temporal_scale = 0.9f * eds->avg_temporal_scale + 0.1f * temporal;
    }

    // 2. Cross-layer modulation
    if (eds->enabled_cross_layer) {
        float resonance = get_layer_resonance(&eds->cross_layer, layer);
        scale *= (1.0f + resonance * 0.3f);
        eds->avg_layer_resonance = 0.9f * eds->avg_layer_resonance + 0.1f * fabsf(resonance);
    }

    // 5. Somatic modulation (before application)
    if (eds->enabled_somatic && body) {
        modulate_delta_by_body(delta, body, &eds->somatic);
    }

    // Apply delta
    float temp[DELTA_RANK];
    memset(temp, 0, sizeof(temp));

    for (int r = 0; r < delta->rank; r++) {
        for (int j = 0; j < delta->in_dim; j++) {
            temp[r] += delta->B[r * delta->in_dim + j] * x[j];
        }
    }

    for (int i = 0; i < delta->out_dim; i++) {
        for (int r = 0; r < delta->rank; r++) {
            out[i] += scale * delta->A[i * delta->rank + r] * temp[r];
        }
    }

    // 3. Track identity drift for contrastive (if enabled)
    if (eds->enabled_contrastive && eds->contrastive.initialized) {
        float drift = compute_identity_drift(&eds->contrastive, out);
        eds->total_drift = 0.95f * eds->total_drift + 0.05f * drift;
    }

    eds->total_applications++;
}

void apply_q_delta_enhanced(DeltaBank* bank, float* q, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body) {
    // Update temporal state
    if (eds->enabled_temporal) {
        update_temporal_state(&eds->temporal, pos);
    }

    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_q_deltas == NULL) continue;

        apply_delta_enhanced_internal(&shard->attn_q_deltas[layer],
                                      q, x, bank->mix[i], layer, pos, eds, body);
    }
}

void apply_k_delta_enhanced(DeltaBank* bank, float* k, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body) {
    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_k_deltas == NULL) continue;

        apply_delta_enhanced_internal(&shard->attn_k_deltas[layer],
                                      k, x, bank->mix[i], layer, pos, eds, body);
    }
}

void apply_v_delta_enhanced(DeltaBank* bank, float* v, float* x, int layer, int dim,
                            int pos, EnhancedDeltaSystem* eds, BodyState* body) {
    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_v_deltas == NULL) continue;

        apply_delta_enhanced_internal(&shard->attn_v_deltas[layer],
                                      v, x, bank->mix[i], layer, pos, eds, body);
    }
}

void update_enhanced_system(EnhancedDeltaSystem* eds, int layer,
                            float* attention_weights, int seq_len) {
    if (!attention_weights || seq_len <= 0) return;

    // Compute attention entropy
    float entropy = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float p = attention_weights[i];
        if (p > 1e-6f) {
            entropy -= p * logf(p);
        }
    }
    float max_entropy = logf((float)seq_len);
    if (max_entropy > 0) {
        entropy /= max_entropy;  // Normalize to [0,1]
    }

    // Update cross-layer interference
    if (eds->enabled_cross_layer) {
        update_interference(&eds->cross_layer, layer, entropy);
    }
}

void print_enhanced_delta_stats(EnhancedDeltaSystem* eds) {
    printf("\n=== Enhanced Delta System ===\n");
    printf("Total applications: %d\n", eds->total_applications);

    // Feature status
    printf("\nEnhancements:\n");
    printf("  Temporal Resonance:    %s (avg scale: %.3f)\n",
           eds->enabled_temporal ? "ON" : "off", eds->avg_temporal_scale);
    printf("  Cross-Layer:           %s (avg resonance: %.3f)\n",
           eds->enabled_cross_layer ? "ON" : "off", eds->avg_layer_resonance);
    printf("  Contrastive Shaping:   %s (drift: %.3f)\n",
           eds->enabled_contrastive ? "ON" : "off", eds->total_drift);
    printf("  Crystallization:       %s\n",
           eds->enabled_crystallization ? "ON" : "off");
    printf("  Somatic Modulation:    %s\n",
           eds->enabled_somatic ? "ON" : "off");

    // Crystallization details
    if (eds->enabled_crystallization) {
        int total_crystallized = 0;
        for (int l = 0; l < N_LAYERS; l++) {
            total_crystallized += eds->crystallization[l].n_crystallized;
        }
        printf("\nCrystallization:\n");
        printf("  Total frozen channels: %d / %d\n",
               total_crystallized, N_LAYERS * DELTA_RANK);
        for (int l = 0; l < N_LAYERS; l++) {
            if (eds->crystallization[l].n_crystallized > 0) {
                printf("  Layer %d: %d channels frozen\n",
                       l, eds->crystallization[l].n_crystallized);
            }
        }
    }

    printf("=============================\n\n");
}

// ============================================================
// Persistence
// ============================================================

int save_enhanced_delta_system(EnhancedDeltaSystem* eds, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    const char magic[] = "EDEL";
    fwrite(magic, 1, 4, f);

    // Save enable flags
    fwrite(&eds->enabled_temporal, sizeof(int), 1, f);
    fwrite(&eds->enabled_cross_layer, sizeof(int), 1, f);
    fwrite(&eds->enabled_contrastive, sizeof(int), 1, f);
    fwrite(&eds->enabled_crystallization, sizeof(int), 1, f);
    fwrite(&eds->enabled_somatic, sizeof(int), 1, f);

    // Save temporal
    fwrite(&eds->temporal, sizeof(TemporalResonance), 1, f);

    // Save cross-layer
    fwrite(&eds->cross_layer, sizeof(CrossLayerState), 1, f);

    // Save contrastive (identity directions)
    fwrite(&eds->contrastive.dim, sizeof(int), 1, f);
    fwrite(&eds->contrastive.initialized, sizeof(int), 1, f);
    if (eds->contrastive.initialized && eds->contrastive.identity_dir) {
        fwrite(eds->contrastive.identity_dir, sizeof(float), eds->contrastive.dim, f);
    }

    // Save crystallization per layer
    for (int l = 0; l < N_LAYERS; l++) {
        CrystallizationState* cs = &eds->crystallization[l];
        fwrite(&cs->n_crystallized, sizeof(int), 1, f);
        fwrite(cs->crystallized_mask, sizeof(int), DELTA_RANK, f);
        if (cs->n_crystallized > 0 && cs->crystal_A) {
            fwrite(cs->crystal_A, sizeof(float), DELTA_RANK * cs->dim, f);
            fwrite(cs->crystal_B, sizeof(float), DELTA_RANK * cs->dim, f);
        }
    }

    // Save somatic
    fwrite(&eds->somatic, sizeof(SomaticModulation), 1, f);

    // Save stats
    fwrite(&eds->total_applications, sizeof(int), 1, f);
    fwrite(&eds->avg_temporal_scale, sizeof(float), 1, f);
    fwrite(&eds->avg_layer_resonance, sizeof(float), 1, f);
    fwrite(&eds->total_drift, sizeof(float), 1, f);

    fclose(f);
    printf("[EnhancedDelta] Saved to %s\n", path);
    return 1;
}

int load_enhanced_delta_system(EnhancedDeltaSystem* eds, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "EDEL", 4) != 0) {
        fclose(f);
        return 0;
    }

    // Load enable flags
    if (fread(&eds->enabled_temporal, sizeof(int), 1, f) != 1 ||
        fread(&eds->enabled_cross_layer, sizeof(int), 1, f) != 1 ||
        fread(&eds->enabled_contrastive, sizeof(int), 1, f) != 1 ||
        fread(&eds->enabled_crystallization, sizeof(int), 1, f) != 1 ||
        fread(&eds->enabled_somatic, sizeof(int), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Load temporal
    if (fread(&eds->temporal, sizeof(TemporalResonance), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Load cross-layer
    if (fread(&eds->cross_layer, sizeof(CrossLayerState), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Load contrastive
    int dim, initialized;
    if (fread(&dim, sizeof(int), 1, f) != 1 ||
        fread(&initialized, sizeof(int), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // SECURITY: validate dim to prevent OOM/corruption
    if (dim < 1 || dim > 8192) {  // Reasonable max for transformer dim
        fprintf(stderr, "[delta_enhanced] corrupt file: invalid dim %d\n", dim);
        fclose(f);
        return 0;
    }

    // Validate dim matches if already initialized
    if (eds->contrastive.dim > 0 && eds->contrastive.dim != dim) {
        fprintf(stderr, "[delta_enhanced] dim mismatch: file=%d, expected=%d\n", dim, eds->contrastive.dim);
        fclose(f);
        return 0;
    }

    if (initialized) {
        if (eds->contrastive.identity_dir == NULL) {
            init_contrastive_forces(&eds->contrastive, dim);
        }
        if (fread(eds->contrastive.identity_dir, sizeof(float), dim, f) != (size_t)dim) {
            fclose(f);
            return 0;
        }
        // Recompute anti-id direction
        for (int i = 0; i < dim; i++) {
            eds->contrastive.anti_id_dir[i] = -eds->contrastive.identity_dir[i];
        }
        eds->contrastive.initialized = 1;
    }

    // Load crystallization per layer
    for (int l = 0; l < N_LAYERS; l++) {
        CrystallizationState* cs = &eds->crystallization[l];
        if (fread(&cs->n_crystallized, sizeof(int), 1, f) != 1 ||
            fread(cs->crystallized_mask, sizeof(int), DELTA_RANK, f) != DELTA_RANK) {
            fclose(f);
            return 0;
        }
        if (cs->n_crystallized > 0) {
            if (cs->crystal_A == NULL) {
                cs->crystal_A = (float*)calloc(DELTA_RANK * cs->dim, sizeof(float));
                cs->crystal_B = (float*)calloc(DELTA_RANK * cs->dim, sizeof(float));
            }
            if (fread(cs->crystal_A, sizeof(float), DELTA_RANK * cs->dim, f) !=
                    (size_t)(DELTA_RANK * cs->dim) ||
                fread(cs->crystal_B, sizeof(float), DELTA_RANK * cs->dim, f) !=
                    (size_t)(DELTA_RANK * cs->dim)) {
                fclose(f);
                return 0;
            }
        }
    }

    // Load somatic
    if (fread(&eds->somatic, sizeof(SomaticModulation), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Load stats
    if (fread(&eds->total_applications, sizeof(int), 1, f) != 1 ||
        fread(&eds->avg_temporal_scale, sizeof(float), 1, f) != 1 ||
        fread(&eds->avg_layer_resonance, sizeof(float), 1, f) != 1 ||
        fread(&eds->total_drift, sizeof(float), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    fclose(f);
    printf("[EnhancedDelta] Loaded from %s\n", path);
    return 1;
}
