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

    mt->pre_trace = (float*)calloc(dim, sizeof(float));
    mt->post_trace = (float*)calloc(dim, sizeof(float));
}

void free_microtrainer(MicroTrainer* mt) {
    if (mt->pre_trace) free(mt->pre_trace);
    if (mt->post_trace) free(mt->post_trace);
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
