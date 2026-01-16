/*
 * delta.c - Dynamic attention deltas implementation
 *
 * Key insight from Stanley: deltas modify ATTENTION
 * They change WHERE the model looks, not WHAT it knows
 */

#include "delta.h"
#include <string.h>

// ============================================================
// Signal extraction
// ============================================================

void init_signals(Signals* sig) {
    sig->arousal = 0.5f;
    sig->entropy = 0.5f;
    sig->tension = 0.3f;
    sig->warmth = 0.6f;
    sig->focus = 0.5f;
    sig->novelty = 0.5f;
    sig->recursion_depth = 0.0f;
    sig->resonance = 0.5f;
}

void extract_signals(Signals* sig, int* tokens, int n_tokens, float* hidden_states) {
    init_signals(sig);

    if (n_tokens == 0) return;

    // Simple heuristics from token patterns
    // (In full version: use hidden states for richer signals)

    int question_marks = 0;
    int exclamation = 0;
    int uppercase = 0;
    int spaces = 0;
    int she_count = 0;

    for (int i = 0; i < n_tokens; i++) {
        int t = tokens[i];
        if (t == '?') question_marks++;
        if (t == '!') exclamation++;
        if (t >= 'A' && t <= 'Z') uppercase++;
        if (t == ' ') spaces++;

        // Check for "she" or "She"
        if (i >= 2) {
            if ((tokens[i-2] == 's' || tokens[i-2] == 'S') &&
                tokens[i-1] == 'h' && tokens[i] == 'e') {
                she_count++;
            }
        }
    }

    float len = (float)n_tokens;

    // Arousal: exclamation, caps
    sig->arousal = fminf(1.0f, (exclamation * 0.3f + uppercase * 0.02f));

    // Entropy: variation in characters
    sig->entropy = fminf(1.0f, (float)spaces / len * 2.0f);

    // Tension: questions
    sig->tension = fminf(1.0f, question_marks * 0.2f);

    // Warmth: emotional indicators (simplified)
    sig->warmth = 0.6f;  // Arianna is warm by default

    // Focus: inverse of length
    sig->focus = fmaxf(0.2f, 1.0f - len / 256.0f);

    // Recursion: self-reference ("she")
    sig->recursion_depth = fminf(1.0f, she_count * 0.2f);

    // Resonance: will be computed from hidden state patterns
    if (hidden_states != NULL) {
        // Placeholder: measure activation magnitude as resonance proxy
        float mag = 0.0f;
        for (int i = 0; i < DIM; i++) {
            mag += hidden_states[i] * hidden_states[i];
        }
        sig->resonance = fminf(1.0f, sqrtf(mag) / 10.0f);
    }
}

// ============================================================
// Low-rank delta operations
// ============================================================

static void init_low_rank_delta(LowRankDelta* d, int out_dim, int in_dim, int rank) {
    d->out_dim = out_dim;
    d->in_dim = in_dim;
    d->rank = rank;
    d->A = (float*)calloc(out_dim * rank, sizeof(float));
    d->B = (float*)calloc(rank * in_dim, sizeof(float));
}

static void free_low_rank_delta(LowRankDelta* d) {
    if (d->A) free(d->A);
    if (d->B) free(d->B);
    d->A = NULL;
    d->B = NULL;
}

// Compute ΔW @ x and add to output
// out += (A @ B) @ x = A @ (B @ x)
static void apply_delta(LowRankDelta* d, float* out, float* x, float scale) {
    if (d->A == NULL || d->B == NULL) return;

    // temp = B @ x  (rank x 1)
    float temp[DELTA_RANK];
    memset(temp, 0, sizeof(temp));

    for (int r = 0; r < d->rank; r++) {
        for (int j = 0; j < d->in_dim; j++) {
            temp[r] += d->B[r * d->in_dim + j] * x[j];
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
// Delta bank
// ============================================================

void init_delta_bank(DeltaBank* bank) {
    memset(bank, 0, sizeof(DeltaBank));
    bank->cache_valid = 0;
}

void free_delta_bank(DeltaBank* bank) {
    for (int i = 0; i < bank->n_shards; i++) {
        free_shard(&bank->shards[i]);
    }
    if (bank->combined_q_delta) free(bank->combined_q_delta);
    if (bank->combined_k_delta) free(bank->combined_k_delta);
    if (bank->combined_v_delta) free(bank->combined_v_delta);
}

// ============================================================
// Mix computation from signals
// ============================================================

void compute_mix(DeltaBank* bank, Signals* sig) {
    if (bank->n_shards == 0) return;

    // Simple signal-to-mix mapping
    // Each shard has implicit "resonance" with certain signal patterns

    float total = 0.0f;
    for (int i = 0; i < bank->n_shards; i++) {
        // Base weight from shard strength
        float w = bank->shards[i].strength;

        // Modulate by signals (placeholder: use name-based heuristics)
        // In full version: each shard stores its "activation pattern"
        w *= (1.0f + sig->resonance * 0.5f);
        w *= (1.0f - sig->tension * 0.3f);  // Less influence under tension

        bank->mix[i] = w;
        total += w;
    }

    // Normalize
    if (total > 0.0f) {
        for (int i = 0; i < bank->n_shards; i++) {
            bank->mix[i] /= total;
        }
    }

    bank->cache_valid = 0;  // Need to recompute combined deltas
}

// ============================================================
// Apply deltas to attention projections
// ============================================================

void apply_q_delta(DeltaBank* bank, float* q, float* x, int layer, int dim) {
    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;  // Skip negligible

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_q_deltas == NULL) continue;

        apply_delta(&shard->attn_q_deltas[layer], q, x, bank->mix[i]);
    }
}

void apply_k_delta(DeltaBank* bank, float* k, float* x, int layer, int dim) {
    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_k_deltas == NULL) continue;

        apply_delta(&shard->attn_k_deltas[layer], k, x, bank->mix[i]);
    }
}

void apply_v_delta(DeltaBank* bank, float* v, float* x, int layer, int dim) {
    for (int i = 0; i < bank->n_shards; i++) {
        if (bank->mix[i] < 0.01f) continue;

        ExperienceShard* shard = &bank->shards[i];
        if (shard->attn_v_deltas == NULL) continue;

        apply_delta(&shard->attn_v_deltas[layer], v, x, bank->mix[i]);
    }
}

// ============================================================
// Shard I/O
// ============================================================

int load_shard(ExperienceShard* shard, const char* path, int n_layers, int dim) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    // Read header
    fread(shard->name, 1, 64, f);
    fread(&shard->strength, sizeof(float), 1, f);
    fread(&shard->n_layers, sizeof(int), 1, f);

    int rank;
    fread(&rank, sizeof(int), 1, f);

    // Allocate deltas
    shard->attn_q_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    shard->attn_k_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));
    shard->attn_v_deltas = (LowRankDelta*)calloc(n_layers, sizeof(LowRankDelta));

    // Read Q deltas
    for (int l = 0; l < n_layers; l++) {
        init_low_rank_delta(&shard->attn_q_deltas[l], dim, dim, rank);
        fread(shard->attn_q_deltas[l].A, sizeof(float), dim * rank, f);
        fread(shard->attn_q_deltas[l].B, sizeof(float), rank * dim, f);
    }

    // Read K deltas
    for (int l = 0; l < n_layers; l++) {
        init_low_rank_delta(&shard->attn_k_deltas[l], dim, dim, rank);
        fread(shard->attn_k_deltas[l].A, sizeof(float), dim * rank, f);
        fread(shard->attn_k_deltas[l].B, sizeof(float), rank * dim, f);
    }

    // Read V deltas
    for (int l = 0; l < n_layers; l++) {
        init_low_rank_delta(&shard->attn_v_deltas[l], dim, dim, rank);
        fread(shard->attn_v_deltas[l].A, sizeof(float), dim * rank, f);
        fread(shard->attn_v_deltas[l].B, sizeof(float), rank * dim, f);
    }

    fclose(f);
    return 0;
}

int save_shard(ExperienceShard* shard, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Write header
    fwrite(shard->name, 1, 64, f);
    fwrite(&shard->strength, sizeof(float), 1, f);
    fwrite(&shard->n_layers, sizeof(int), 1, f);

    int rank = shard->attn_q_deltas[0].rank;
    fwrite(&rank, sizeof(int), 1, f);

    // Write deltas
    for (int l = 0; l < shard->n_layers; l++) {
        LowRankDelta* d = &shard->attn_q_deltas[l];
        fwrite(d->A, sizeof(float), d->out_dim * d->rank, f);
        fwrite(d->B, sizeof(float), d->rank * d->in_dim, f);
    }
    for (int l = 0; l < shard->n_layers; l++) {
        LowRankDelta* d = &shard->attn_k_deltas[l];
        fwrite(d->A, sizeof(float), d->out_dim * d->rank, f);
        fwrite(d->B, sizeof(float), d->rank * d->in_dim, f);
    }
    for (int l = 0; l < shard->n_layers; l++) {
        LowRankDelta* d = &shard->attn_v_deltas[l];
        fwrite(d->A, sizeof(float), d->out_dim * d->rank, f);
        fwrite(d->B, sizeof(float), d->rank * d->in_dim, f);
    }

    fclose(f);
    return 0;
}

void free_shard(ExperienceShard* shard) {
    if (shard->attn_q_deltas) {
        for (int l = 0; l < shard->n_layers; l++) {
            free_low_rank_delta(&shard->attn_q_deltas[l]);
        }
        free(shard->attn_q_deltas);
    }
    if (shard->attn_k_deltas) {
        for (int l = 0; l < shard->n_layers; l++) {
            free_low_rank_delta(&shard->attn_k_deltas[l]);
        }
        free(shard->attn_k_deltas);
    }
    if (shard->attn_v_deltas) {
        for (int l = 0; l < shard->n_layers; l++) {
            free_low_rank_delta(&shard->attn_v_deltas[l]);
        }
        free(shard->attn_v_deltas);
    }
}

int add_shard(DeltaBank* bank, const char* path, int n_layers, int dim) {
    if (bank->n_shards >= MAX_SHARDS) return -1;

    int idx = bank->n_shards;
    if (load_shard(&bank->shards[idx], path, n_layers, dim) != 0) {
        return -1;
    }

    bank->mix[idx] = 1.0f / (bank->n_shards + 1);  // Initial equal mix
    bank->n_shards++;

    return idx;
}

// ============================================================
// Microtrainer - Hebbian online learning
// ============================================================

void init_microtrainer(MicroTrainer* mt, int dim) {
    mt->learning_rate = 0.001f;
    mt->momentum = 0.9f;
    mt->decay = 0.999f;

    // Contrastive learning params (from lora.c)
    mt->push = 1.0f;    // Boost target
    mt->pull = 0.5f;    // Suppress competitors
    mt->topk = 3;       // Top 3 competitors

    // Deterministic noise channel
    mt->seed = 0xA17A11u;  // "ARIANNA" in hex-ish
    mt->u = NULL;          // Lazy allocated
    mt->dy = NULL;         // Lazy allocated

    mt->dim = dim;
    mt->vocab_size = 0;    // Set later

    mt->pre_trace = (float*)calloc(dim, sizeof(float));
    mt->post_trace = (float*)calloc(dim, sizeof(float));
}

void free_microtrainer(MicroTrainer* mt) {
    if (mt->pre_trace) free(mt->pre_trace);
    if (mt->post_trace) free(mt->post_trace);
    if (mt->u) free(mt->u);
    if (mt->dy) free(mt->dy);
}

/*
 * Hebbian update: "neurons that fire together, wire together"
 *
 * ΔW ∝ pre * post * reward
 *
 * For low-rank: update A and B separately
 * ΔA ∝ post * (B @ pre) * reward
 * ΔB ∝ (A^T @ post) * pre * reward
 */
void micro_update(MicroTrainer* mt, LowRankDelta* delta,
                  float* pre, float* post, float reward) {
    if (delta->A == NULL || delta->B == NULL) return;
    if (fabsf(reward) < 0.001f) return;  // Skip tiny rewards

    float lr = mt->learning_rate * reward;

    // Update traces with momentum
    for (int i = 0; i < delta->in_dim && i < DIM; i++) {
        mt->pre_trace[i] = mt->momentum * mt->pre_trace[i] + (1.0f - mt->momentum) * pre[i];
    }
    for (int i = 0; i < delta->out_dim && i < DIM; i++) {
        mt->post_trace[i] = mt->momentum * mt->post_trace[i] + (1.0f - mt->momentum) * post[i];
    }

    // Compute B @ pre_trace
    float b_pre[DELTA_RANK];
    memset(b_pre, 0, sizeof(b_pre));
    for (int r = 0; r < delta->rank; r++) {
        for (int j = 0; j < delta->in_dim; j++) {
            b_pre[r] += delta->B[r * delta->in_dim + j] * mt->pre_trace[j];
        }
    }

    // Update A: ΔA[i,r] = lr * post_trace[i] * b_pre[r]
    for (int i = 0; i < delta->out_dim; i++) {
        for (int r = 0; r < delta->rank; r++) {
            delta->A[i * delta->rank + r] += lr * mt->post_trace[i] * b_pre[r];
            // Apply decay
            delta->A[i * delta->rank + r] *= mt->decay;
        }
    }

    // Compute A^T @ post_trace
    float at_post[DELTA_RANK];
    memset(at_post, 0, sizeof(at_post));
    for (int r = 0; r < delta->rank; r++) {
        for (int i = 0; i < delta->out_dim; i++) {
            at_post[r] += delta->A[i * delta->rank + r] * mt->post_trace[i];
        }
    }

    // Update B: ΔB[r,j] = lr * at_post[r] * pre_trace[j]
    for (int r = 0; r < delta->rank; r++) {
        for (int j = 0; j < delta->in_dim; j++) {
            delta->B[r * delta->in_dim + j] += lr * at_post[r] * mt->pre_trace[j];
            // Apply decay
            delta->B[r * delta->in_dim + j] *= mt->decay;
        }
    }
}

// ============================================================
// Notorch Plasticity (ported from lang/lora.c)
// "This is NOT gradient descent. It's plasticity."
// ============================================================

// Tiny deterministic RNG (xorshift32)
static unsigned int xorshift32(unsigned int* s) {
    unsigned int x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float frand01(unsigned int* s) {
    return (xorshift32(s) & 0xFFFFFF) / 16777216.0f;
}

static float frandn(unsigned int* s) {
    // Box-Muller
    float u1 = fmaxf(1e-6f, fminf(frand01(s), 1.0f));
    float u2 = frand01(s);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// Find index of max prob excluding target
static int argmax_excluding(const float* probs, int n, int exclude) {
    int imax = (exclude == 0 ? 1 : 0);
    if (imax >= n) return -1;
    float pmax = probs[imax];
    for (int i = 0; i < n; i++) {
        if (i == exclude) continue;
        if (probs[i] > pmax) { pmax = probs[i]; imax = i; }
    }
    return imax;
}

/*
 * Build dy from probs: push target, pull competitors
 *
 * Strategy:
 *  - dy[target] += push
 *  - find top competitors and push them down: dy[comp] -= pull / K
 */
void build_dy_from_probs(MicroTrainer* mt, float* dy_out,
                         const float* probs, int vocab_size,
                         int target_id) {
    if (!dy_out || !probs || vocab_size <= 0) return;
    if (target_id < 0 || target_id >= vocab_size) return;

    // Zero output
    memset(dy_out, 0, vocab_size * sizeof(float));

    // Main push: boost target
    dy_out[target_id] += mt->push;

    // Pull: suppress competitors
    if (mt->topk <= 0) {
        // Just suppress strongest competitor
        int comp = argmax_excluding(probs, vocab_size, target_id);
        if (comp >= 0) dy_out[comp] -= mt->pull;
    } else {
        // Top-K suppression
        float each_pull = mt->pull / (float)mt->topk;
        for (int k = 0; k < mt->topk; k++) {
            int comp = argmax_excluding(probs, vocab_size, target_id);
            if (comp >= 0) {
                dy_out[comp] -= each_pull;
            }
        }
    }
}

/*
 * Notorch step: plasticity without backprop
 *
 * The idea:
 * 1) We have input x and desired delta direction dy
 * 2) We pick a rank-channel vector u (deterministic noise)
 * 3) Update factors:
 *      A += lr * (x ⊗ u)
 *      B += lr * (u ⊗ dy)
 * 4) Gentle decay to prevent runaway
 */
void notorch_step(MicroTrainer* mt, LowRankDelta* delta,
                  const float* x, const float* dy, float signal) {
    if (!mt || !delta || !x || !dy) return;
    if (delta->A == NULL || delta->B == NULL) return;

    // Clamp signal
    float g = fmaxf(-2.0f, fminf(signal, 2.0f));

    // Allocate u if not already
    if (mt->u == NULL) {
        mt->u = (float*)calloc(DELTA_RANK, sizeof(float));
    }

    // Build u: deterministic noise modulated by signal strength
    // Stronger signal -> cleaner channel (less noise)
    for (int r = 0; r < delta->rank; r++) {
        float n = frandn(&mt->seed);
        float k = 0.35f + 0.65f * (1.0f - fabsf(g));
        mt->u[r] = n * k;
    }

    float lr = mt->learning_rate;

    // A[i,r] += lr * x[i] * u[r] * g
    for (int i = 0; i < delta->in_dim; i++) {
        float xi = x[i] * lr * g;
        for (int r = 0; r < delta->rank; r++) {
            delta->A[i * delta->rank + r] += xi * mt->u[r];
        }
    }

    // B[r,j] += lr * u[r] * dy[j] * g
    for (int r = 0; r < delta->rank; r++) {
        float ur = mt->u[r] * lr * g;
        for (int j = 0; j < delta->out_dim; j++) {
            delta->B[r * delta->out_dim + j] += ur * dy[j];
        }
    }

    // Gentle decay
    if (mt->decay > 0.0f && mt->decay < 1.0f) {
        float d = mt->decay;
        for (int i = 0; i < delta->in_dim * delta->rank; i++) {
            delta->A[i] *= d;
        }
        for (int i = 0; i < delta->rank * delta->out_dim; i++) {
            delta->B[i] *= d;
        }
    }
}

/*
 * Experience step: one-call wrapper for notorch learning
 * Builds dy from probs internally, then applies notorch_step
 *
 * This is the "breathing" interface:
 *   model.forward() → probs
 *   experience_step(probs, target, signal) → personality update
 */
void experience_step(MicroTrainer* mt, LowRankDelta* delta,
                     const float* x, const float* probs,
                     int target_id, float signal) {
    if (!mt || !delta || !x || !probs) return;
    if (target_id < 0 || target_id >= delta->out_dim) return;

    // Allocate dy if not already
    if (mt->dy == NULL) {
        mt->dy = (float*)calloc(delta->out_dim, sizeof(float));
        mt->vocab_size = delta->out_dim;
    }

    // Build dy from probs
    build_dy_from_probs(mt, mt->dy, probs, delta->out_dim, target_id);

    // Apply notorch step
    notorch_step(mt, delta, x, mt->dy, signal);
}

/*
 * Soft reset: gradual forgetting (scale down instead of zeroing)
 * keep_ratio: 0.0 = full reset, 1.0 = no change
 */
void soft_reset_delta(LowRankDelta* delta, float keep_ratio) {
    if (!delta || delta->A == NULL || delta->B == NULL) return;

    float k = fmaxf(0.0f, fminf(keep_ratio, 1.0f));

    for (int i = 0; i < delta->in_dim * delta->rank; i++) {
        delta->A[i] *= k;
    }
    for (int i = 0; i < delta->rank * delta->out_dim; i++) {
        delta->B[i] *= k;
    }
}

/*
 * Get delta norm (for monitoring)
 */
float get_delta_norm(LowRankDelta* delta) {
    if (!delta || delta->A == NULL || delta->B == NULL) return 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < delta->in_dim * delta->rank; i++) {
        sum += delta->A[i] * delta->A[i];
    }
    for (int i = 0; i < delta->rank * delta->out_dim; i++) {
        sum += delta->B[i] * delta->B[i];
    }

    return sqrtf(sum);
}

/*
 * Clamp delta to prevent weight explosion
 */
void clamp_delta(LowRankDelta* delta, float max_norm) {
    if (!delta || max_norm <= 0.0f) return;

    float norm = get_delta_norm(delta);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        soft_reset_delta(delta, scale);
    }
}

// ============================================================
// Notorch Microlearning Revolution
// "Pure C plasticity without PyTorch compromise"
// ============================================================

/*
 * 1. Resonance-Gated Plasticity
 * Learn more when aligned with identity, less when drifting
 */
void experience_step_gated(MicroTrainer* mt, LowRankDelta* delta,
                           const float* x, const float* probs,
                           int target_id, float signal,
                           const float* identity_embedding, int dim) {
    if (!mt || !delta || !x || !probs || !identity_embedding) return;
    if (dim <= 0) return;

    // Compute resonance with identity (cosine similarity)
    float dot = 0.0f, norm_x = 0.0f, norm_id = 0.0f;
    int compute_dim = (dim < delta->in_dim) ? dim : delta->in_dim;

    for (int i = 0; i < compute_dim; i++) {
        dot += x[i] * identity_embedding[i];
        norm_x += x[i] * x[i];
        norm_id += identity_embedding[i] * identity_embedding[i];
    }

    float denom = sqrtf(norm_x * norm_id) + 1e-6f;
    float resonance = dot / denom;

    // Gate: learn more when aligned with identity, less when drifting
    // resonance in [-1, 1], we want gate in [0.3, 1.0]
    // If drifting (resonance < 0), almost no learning
    // If aligned (resonance > 0.5), full learning
    float gate = 0.3f + 0.7f * fmaxf(0.0f, resonance);

    // Modulated signal
    float gated_signal = signal * gate;

    // Standard experience step with gated signal
    experience_step(mt, delta, x, probs, target_id, gated_signal);
}

/*
 * 2. Adaptive Push/Pull
 * High confidence = more pull (suppress competitors)
 * Low confidence = more push (boost target)
 */
void set_adaptive_push_pull(MicroTrainer* mt, const float* probs,
                            int vocab_size, int target_id) {
    if (!mt || !probs || vocab_size <= 0) return;
    if (target_id < 0 || target_id >= vocab_size) return;

    float target_prob = probs[target_id];

    // Find max competitor prob
    float max_comp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (i != target_id && probs[i] > max_comp) {
            max_comp = probs[i];
        }
    }

    // Confidence: how much higher is target than competitors?
    float confidence = target_prob - max_comp;

    // Adaptive push/pull
    // Low confidence (target weak): push hard, pull soft
    // High confidence (target strong): push soft, pull hard
    if (confidence < 0.0f) {
        // Target is losing - boost it hard
        mt->push = 1.5f;
        mt->pull = 0.3f;
    } else if (confidence < 0.3f) {
        // Close race - balanced
        mt->push = 1.0f;
        mt->pull = 0.5f;
    } else {
        // Target winning - suppress competitors
        mt->push = 0.5f;
        mt->pull = 1.0f;
    }
}

/*
 * 3. Quality-Weighted Signal
 * Good generations teach more, stuck/boring teach less
 */
float compute_quality_weight(float quality, float stuck, float boredom) {
    // Quality in [0, 1], higher = better
    // Stuck in [0, 1], higher = more stuck
    // Boredom in [0, 1], higher = more bored

    // Base weight from quality: quality 0.9 -> 1.35, quality 0.5 -> 0.75
    float weight = 0.3f + 1.2f * quality;

    // Reduce weight when stuck (repetitive patterns shouldn't reinforce)
    weight *= (1.0f - 0.4f * stuck);

    // Slightly reduce when bored (low novelty shouldn't dominate)
    weight *= (1.0f - 0.15f * boredom);

    return fmaxf(0.1f, fminf(weight, 1.5f));
}

/*
 * 4. Spectral Channel Freezing
 * Returns 1 if channel should be frozen based on its energy
 */
int should_freeze_channel(LowRankDelta* delta, int channel, float threshold) {
    if (!delta || !delta->A || channel < 0 || channel >= delta->rank) {
        return 0;
    }

    // Compute channel energy (L2 norm of A column)
    float energy = 0.0f;
    for (int i = 0; i < delta->in_dim; i++) {
        float val = delta->A[i * delta->rank + channel];
        energy += val * val;
    }
    energy = sqrtf(energy);

    return (energy > threshold) ? 1 : 0;
}

/*
 * 5. Experience Consolidation
 * Average frozen channels into core experience
 */
void consolidate_experience(LowRankDelta* delta, LowRankDelta* core,
                            int* frozen_mask, int n_frozen) {
    if (!delta || !core || !frozen_mask || n_frozen <= 0) return;
    if (!delta->A || !delta->B || !core->A || !core->B) return;

    // For each frozen channel, blend into core
    float blend = 0.3f;  // 30% of frozen pattern goes to core

    for (int r = 0; r < delta->rank && r < core->rank; r++) {
        if (!frozen_mask[r]) continue;

        // Blend A column
        for (int i = 0; i < delta->in_dim && i < core->in_dim; i++) {
            float delta_val = delta->A[i * delta->rank + r];
            float core_val = core->A[i * core->rank + r];
            core->A[i * core->rank + r] = core_val * (1.0f - blend) + delta_val * blend;
        }

        // Blend B row
        for (int j = 0; j < delta->out_dim && j < core->out_dim; j++) {
            float delta_val = delta->B[r * delta->out_dim + j];
            float core_val = core->B[r * core->out_dim + j];
            core->B[r * core->out_dim + j] = core_val * (1.0f - blend) + delta_val * blend;
        }
    }
}
