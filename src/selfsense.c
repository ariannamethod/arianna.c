/*
 * selfsense.c - Self-sensing from hidden states
 *
 * "The model feels itself from the inside"
 */

#include "selfsense.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================
// Initialization
// ============================================================

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

static float xavier_init(int fan_in, int fan_out) {
    float scale = sqrtf(6.0f / (fan_in + fan_out));
    return (randf() * 2.0f - 1.0f) * scale;
}

void init_selfsense(SelfSense* ss, int dim) {
    memset(ss, 0, sizeof(SelfSense));
    ss->dim = dim;

    // Allocate MLP weights
    ss->mlp.w1 = (float*)calloc(SELFSENSE_HIDDEN_DIM * dim, sizeof(float));
    ss->mlp.b1 = (float*)calloc(SELFSENSE_HIDDEN_DIM, sizeof(float));
    ss->mlp.w2 = (float*)calloc(SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM, sizeof(float));
    ss->mlp.b2 = (float*)calloc(SELFSENSE_OUTPUT_DIM, sizeof(float));

    // Xavier initialization
    for (int i = 0; i < SELFSENSE_HIDDEN_DIM * dim; i++) {
        ss->mlp.w1[i] = xavier_init(dim, SELFSENSE_HIDDEN_DIM);
    }
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM; i++) {
        ss->mlp.w2[i] = xavier_init(SELFSENSE_HIDDEN_DIM, SELFSENSE_OUTPUT_DIM);
    }

    // Initialize biases to small positive values (signals start slightly active)
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        ss->mlp.b2[i] = 0.1f;
    }

    // Initialize state
    ss->state.ema_momentum = 0.85f;  // Moderate smoothing
    ss->state.history_pos = 0;
    ss->state.history_count = 0;

    // Initialize EMA to neutral values
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        ss->state.ema_signals[i] = 0.5f;
    }

    // Allocate identity embeddings
    ss->identity.identity_embedding = (float*)calloc(dim, sizeof(float));
    ss->identity.warmth_direction = (float*)calloc(dim, sizeof(float));
    ss->identity.cold_direction = (float*)calloc(dim, sizeof(float));
    ss->identity.initialized = 0;

    // Learning params
    ss->learning_rate = 0.001f;
    ss->observations = 0;
    ss->running_loss = 0.0f;

    ss->initialized = 1;
}

void free_selfsense(SelfSense* ss) {
    if (!ss->initialized) return;

    free(ss->mlp.w1);
    free(ss->mlp.b1);
    free(ss->mlp.w2);
    free(ss->mlp.b2);
    free(ss->identity.identity_embedding);
    free(ss->identity.warmth_direction);
    free(ss->identity.cold_direction);

    memset(ss, 0, sizeof(SelfSense));
}

// ============================================================
// MLP Forward Pass
// ============================================================

static float tanh_activation(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2x = expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

static float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void mlp_forward(SelfSenseMLP* mlp, float* input, int dim) {
    // First layer: input → hidden (tanh activation)
    for (int h = 0; h < SELFSENSE_HIDDEN_DIM; h++) {
        float sum = mlp->b1[h];
        for (int i = 0; i < dim; i++) {
            sum += mlp->w1[h * dim + i] * input[i];
        }
        mlp->hidden[h] = tanh_activation(sum);
    }

    // Second layer: hidden → output (sigmoid for 0-1 range)
    for (int o = 0; o < SELFSENSE_OUTPUT_DIM; o++) {
        float sum = mlp->b2[o];
        for (int h = 0; h < SELFSENSE_HIDDEN_DIM; h++) {
            sum += mlp->w2[o * SELFSENSE_HIDDEN_DIM + h] * mlp->hidden[h];
        }
        mlp->raw_output[o] = sigmoid(sum);
    }
}

// ============================================================
// Analysis Functions
// ============================================================

float compute_activation_entropy(float* hidden, int dim) {
    // Treat activations as a probability distribution
    // Higher entropy = more uniform = more chaotic
    float sum = 0.0f;
    float max_val = -1e9f;

    // Find max for numerical stability
    for (int i = 0; i < dim; i++) {
        float val = fabsf(hidden[i]);
        if (val > max_val) max_val = val;
    }

    if (max_val < 1e-6f) return 0.0f;

    // Softmax-like normalization
    for (int i = 0; i < dim; i++) {
        float val = fabsf(hidden[i]) / max_val;
        sum += val;
    }

    if (sum < 1e-6f) return 0.0f;

    // Compute entropy
    float entropy = 0.0f;
    for (int i = 0; i < dim; i++) {
        float p = fabsf(hidden[i]) / max_val / sum;
        if (p > 1e-6f) {
            entropy -= p * logf(p);
        }
    }

    // Normalize by max possible entropy (uniform distribution)
    float max_entropy = logf((float)dim);
    return fminf(1.0f, entropy / max_entropy);
}

float compute_layer_tension(float* pre_layer, float* post_layer, int dim) {
    // How much did the representation change?
    // High tension = big transformation
    if (pre_layer == NULL || post_layer == NULL) return 0.5f;

    float diff_sq = 0.0f;
    float pre_mag = 0.0f;

    for (int i = 0; i < dim; i++) {
        float d = post_layer[i] - pre_layer[i];
        diff_sq += d * d;
        pre_mag += pre_layer[i] * pre_layer[i];
    }

    if (pre_mag < 1e-6f) return 0.5f;

    float relative_change = sqrtf(diff_sq) / sqrtf(pre_mag);
    return fminf(1.0f, relative_change);
}

float compute_attention_focus(float* attention_weights, int seq_len) {
    // Gini coefficient of attention - higher = more focused
    if (attention_weights == NULL || seq_len <= 1) return 0.5f;

    // Sort attention weights (simple bubble sort, seq_len is small)
    float sorted[MAX_SEQ_LEN];
    memcpy(sorted, attention_weights, seq_len * sizeof(float));

    for (int i = 0; i < seq_len - 1; i++) {
        for (int j = 0; j < seq_len - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float tmp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = tmp;
            }
        }
    }

    // Compute Gini coefficient
    float sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += sorted[i];
        weighted_sum += (i + 1) * sorted[i];
    }

    if (sum < 1e-6f) return 0.5f;

    float gini = (2.0f * weighted_sum) / (seq_len * sum) - (seq_len + 1.0f) / seq_len;
    return fminf(1.0f, fmaxf(0.0f, gini));
}

float compute_identity_alignment(float* hidden, float* identity, int dim) {
    // Cosine similarity between hidden state and identity embedding
    if (identity == NULL) return 0.5f;

    float dot = 0.0f;
    float mag_h = 0.0f;
    float mag_i = 0.0f;

    for (int i = 0; i < dim; i++) {
        dot += hidden[i] * identity[i];
        mag_h += hidden[i] * hidden[i];
        mag_i += identity[i] * identity[i];
    }

    if (mag_h < 1e-6f || mag_i < 1e-6f) return 0.5f;

    float cosine = dot / (sqrtf(mag_h) * sqrtf(mag_i));
    // Map from [-1, 1] to [0, 1]
    return (cosine + 1.0f) / 2.0f;
}

// ============================================================
// Main Extraction
// ============================================================

void selfsense_extract(SelfSense* ss, float* hidden_states, Signals* out_signals) {
    if (!ss->initialized || hidden_states == NULL) return;

    // Run MLP forward pass
    mlp_forward(&ss->mlp, hidden_states, ss->dim);

    // Store raw output
    memcpy(ss->last_raw, ss->mlp.raw_output, SELFSENSE_OUTPUT_DIM * sizeof(float));

    // Compute additional metrics directly from hidden states
    float entropy = compute_activation_entropy(hidden_states, ss->dim);
    float resonance = 0.5f;

    if (ss->identity.initialized) {
        resonance = compute_identity_alignment(hidden_states,
                                               ss->identity.identity_embedding,
                                               ss->dim);
    }

    // Blend MLP output with direct metrics
    // MLP learns the mapping, but we also inject some direct observations
    float blended[SELFSENSE_OUTPUT_DIM];
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        blended[i] = ss->mlp.raw_output[i];
    }

    // Override entropy with direct computation (MLP can't easily learn this)
    blended[SIG_ENTROPY] = entropy * 0.7f + ss->mlp.raw_output[SIG_ENTROPY] * 0.3f;

    // Override resonance with identity alignment
    blended[SIG_RESONANCE] = resonance * 0.6f + ss->mlp.raw_output[SIG_RESONANCE] * 0.4f;

    // Apply EMA smoothing
    float mom = ss->state.ema_momentum;
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        ss->state.ema_signals[i] = mom * ss->state.ema_signals[i] +
                                   (1.0f - mom) * blended[i];
    }

    // Store in history
    int pos = ss->state.history_pos;
    memcpy(ss->state.history[pos], ss->state.ema_signals,
           SELFSENSE_OUTPUT_DIM * sizeof(float));
    ss->state.history_pos = (pos + 1) % SELFSENSE_HISTORY_LEN;
    if (ss->state.history_count < SELFSENSE_HISTORY_LEN) {
        ss->state.history_count++;
    }

    // Update variance and trend
    if (ss->state.history_count >= 8) {
        for (int s = 0; s < SELFSENSE_OUTPUT_DIM; s++) {
            // Compute variance over last 8 samples
            float mean = 0.0f;
            for (int h = 0; h < 8; h++) {
                int idx = (pos - h - 1 + SELFSENSE_HISTORY_LEN) % SELFSENSE_HISTORY_LEN;
                mean += ss->state.history[idx][s];
            }
            mean /= 8.0f;

            float var = 0.0f;
            for (int h = 0; h < 8; h++) {
                int idx = (pos - h - 1 + SELFSENSE_HISTORY_LEN) % SELFSENSE_HISTORY_LEN;
                float d = ss->state.history[idx][s] - mean;
                var += d * d;
            }
            ss->state.signal_variance[s] = var / 8.0f;

            // Compute trend (recent 4 vs previous 4)
            float recent = 0.0f, older = 0.0f;
            for (int h = 0; h < 4; h++) {
                int idx_r = (pos - h - 1 + SELFSENSE_HISTORY_LEN) % SELFSENSE_HISTORY_LEN;
                int idx_o = (pos - h - 5 + SELFSENSE_HISTORY_LEN) % SELFSENSE_HISTORY_LEN;
                recent += ss->state.history[idx_r][s];
                older += ss->state.history[idx_o][s];
            }
            ss->state.signal_trend[s] = (recent - older) / 4.0f;
        }
    }

    // Map to Signals struct (for compatibility with existing mood routing)
    if (out_signals != NULL) {
        out_signals->arousal = ss->state.ema_signals[SIG_AROUSAL];
        out_signals->entropy = ss->state.ema_signals[SIG_ENTROPY];
        out_signals->tension = ss->state.ema_signals[SIG_TENSION];
        out_signals->warmth = ss->state.ema_signals[SIG_WARMTH];
        out_signals->focus = ss->state.ema_signals[SIG_FOCUS];
        out_signals->recursion_depth = ss->state.ema_signals[SIG_RECURSION];
        out_signals->resonance = ss->state.ema_signals[SIG_RESONANCE];
        // novelty maps to... we don't have a direct field, but we can use it
    }

    ss->observations++;
}

// ============================================================
// Identity Setup
// ============================================================

void selfsense_set_identity(SelfSense* ss, float* identity_embedding) {
    if (!ss->initialized || identity_embedding == NULL) return;

    memcpy(ss->identity.identity_embedding, identity_embedding,
           ss->dim * sizeof(float));
    ss->identity.initialized = 1;
}

void selfsense_compute_identity_from_tokens(SelfSense* ss, Transformer* t,
                                            int* tokens, int n_tokens) {
    if (!ss->initialized || n_tokens == 0) return;

    // Average the token embeddings to get identity embedding
    memset(ss->identity.identity_embedding, 0, ss->dim * sizeof(float));

    for (int i = 0; i < n_tokens; i++) {
        int tok = tokens[i] & 0xFF;
        float* emb = t->weights.token_embedding + tok * ss->dim;
        for (int d = 0; d < ss->dim; d++) {
            ss->identity.identity_embedding[d] += emb[d];
        }
    }

    // Normalize
    for (int d = 0; d < ss->dim; d++) {
        ss->identity.identity_embedding[d] /= n_tokens;
    }

    ss->identity.initialized = 1;
    printf("[SelfSense] Identity computed from %d tokens\n", n_tokens);
}

// ============================================================
// Learning
// ============================================================

void selfsense_learn(SelfSense* ss, float quality) {
    if (!ss->initialized) return;

    // Quality signal: how good was the generation?
    // Use this to adjust the MLP to produce signals that correlate with quality

    // Simple approach: if quality is high, reinforce current signal pattern
    // If low, push away from current pattern

    float target[SELFSENSE_OUTPUT_DIM];
    float error = 0.0f;

    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        // Target is current signal if quality high, opposite if low
        if (quality > 0.5f) {
            target[i] = ss->last_raw[i];  // Reinforce
        } else {
            target[i] = 1.0f - ss->last_raw[i];  // Push away
        }
        float d = target[i] - ss->last_raw[i];
        error += d * d;
    }

    // Update running loss
    ss->running_loss = 0.95f * ss->running_loss + 0.05f * error;

    // Gradient descent on MLP (simplified)
    float lr = ss->learning_rate * fabsf(quality - 0.5f) * 2.0f;  // Scale by deviation

    // Update output layer
    for (int o = 0; o < SELFSENSE_OUTPUT_DIM; o++) {
        float delta = (target[o] - ss->last_raw[o]) * lr;
        ss->mlp.b2[o] += delta;

        for (int h = 0; h < SELFSENSE_HIDDEN_DIM; h++) {
            ss->mlp.w2[o * SELFSENSE_HIDDEN_DIM + h] += delta * ss->mlp.hidden[h];
        }
    }
}

// ============================================================
// History Analysis
// ============================================================

int selfsense_detect_stuck(SelfSense* ss) {
    if (ss->state.history_count < 16) return 0;

    // Stuck = low variance across all signals
    float total_var = 0.0f;
    for (int i = 0; i < SELFSENSE_OUTPUT_DIM; i++) {
        total_var += ss->state.signal_variance[i];
    }

    return total_var < 0.01f;  // Very low variance = stuck
}

int selfsense_detect_spiral(SelfSense* ss) {
    if (ss->state.history_count < 16) return 0;

    // Spiral = recursion and tension both increasing
    float recursion_trend = ss->state.signal_trend[SIG_RECURSION];
    float tension_trend = ss->state.signal_trend[SIG_TENSION];

    return (recursion_trend > 0.02f && tension_trend > 0.02f);
}

float selfsense_get_trend(SelfSense* ss, int signal_idx) {
    if (signal_idx < 0 || signal_idx >= SELFSENSE_OUTPUT_DIM) return 0.0f;
    return ss->state.signal_trend[signal_idx];
}

// ============================================================
// Persistence
// ============================================================

int save_selfsense(SelfSense* ss, const char* path) {
    if (!ss->initialized) return 0;

    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    // Magic
    fwrite("SELF", 1, 4, f);

    // Config
    fwrite(&ss->dim, sizeof(int), 1, f);
    fwrite(&ss->learning_rate, sizeof(float), 1, f);
    fwrite(&ss->observations, sizeof(int), 1, f);
    fwrite(&ss->running_loss, sizeof(float), 1, f);

    // MLP weights
    fwrite(ss->mlp.w1, sizeof(float), SELFSENSE_HIDDEN_DIM * ss->dim, f);
    fwrite(ss->mlp.b1, sizeof(float), SELFSENSE_HIDDEN_DIM, f);
    fwrite(ss->mlp.w2, sizeof(float), SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM, f);
    fwrite(ss->mlp.b2, sizeof(float), SELFSENSE_OUTPUT_DIM, f);

    // EMA state
    fwrite(&ss->state.ema_momentum, sizeof(float), 1, f);
    fwrite(ss->state.ema_signals, sizeof(float), SELFSENSE_OUTPUT_DIM, f);

    // Identity
    fwrite(&ss->identity.initialized, sizeof(int), 1, f);
    if (ss->identity.initialized) {
        fwrite(ss->identity.identity_embedding, sizeof(float), ss->dim, f);
    }

    fclose(f);
    printf("[SelfSense] Saved to %s (%d observations)\n", path, ss->observations);
    return 1;
}

int load_selfsense(SelfSense* ss, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    // Check magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SELF", 4) != 0) {
        fclose(f);
        return 0;
    }

    int dim;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Initialize with correct dim
    init_selfsense(ss, dim);

    // Read config
    if (fread(&ss->learning_rate, sizeof(float), 1, f) != 1 ||
        fread(&ss->observations, sizeof(int), 1, f) != 1 ||
        fread(&ss->running_loss, sizeof(float), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Read MLP weights
    if (fread(ss->mlp.w1, sizeof(float), SELFSENSE_HIDDEN_DIM * dim, f) !=
            (size_t)(SELFSENSE_HIDDEN_DIM * dim) ||
        fread(ss->mlp.b1, sizeof(float), SELFSENSE_HIDDEN_DIM, f) != SELFSENSE_HIDDEN_DIM ||
        fread(ss->mlp.w2, sizeof(float), SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM, f) !=
            SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM ||
        fread(ss->mlp.b2, sizeof(float), SELFSENSE_OUTPUT_DIM, f) != SELFSENSE_OUTPUT_DIM) {
        fclose(f);
        return 0;
    }

    // Read EMA state
    if (fread(&ss->state.ema_momentum, sizeof(float), 1, f) != 1 ||
        fread(ss->state.ema_signals, sizeof(float), SELFSENSE_OUTPUT_DIM, f) !=
            SELFSENSE_OUTPUT_DIM) {
        fclose(f);
        return 0;
    }

    // Read identity
    if (fread(&ss->identity.initialized, sizeof(int), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    if (ss->identity.initialized) {
        if (fread(ss->identity.identity_embedding, sizeof(float), dim, f) != (size_t)dim) {
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    printf("[SelfSense] Loaded: %d observations, loss=%.4f\n",
           ss->observations, ss->running_loss);
    return 1;
}

// ============================================================
// Debug
// ============================================================

void print_selfsense_signals(SelfSense* ss) {
    if (!ss->initialized) {
        printf("SelfSense: not initialized\n");
        return;
    }

    printf("\n=== SelfSense Signals ===\n");
    printf("  arousal:   %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_AROUSAL],
           ss->last_raw[SIG_AROUSAL],
           ss->state.signal_trend[SIG_AROUSAL]);
    printf("  entropy:   %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_ENTROPY],
           ss->last_raw[SIG_ENTROPY],
           ss->state.signal_trend[SIG_ENTROPY]);
    printf("  tension:   %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_TENSION],
           ss->last_raw[SIG_TENSION],
           ss->state.signal_trend[SIG_TENSION]);
    printf("  warmth:    %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_WARMTH],
           ss->last_raw[SIG_WARMTH],
           ss->state.signal_trend[SIG_WARMTH]);
    printf("  focus:     %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_FOCUS],
           ss->last_raw[SIG_FOCUS],
           ss->state.signal_trend[SIG_FOCUS]);
    printf("  recursion: %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_RECURSION],
           ss->last_raw[SIG_RECURSION],
           ss->state.signal_trend[SIG_RECURSION]);
    printf("  resonance: %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_RESONANCE],
           ss->last_raw[SIG_RESONANCE],
           ss->state.signal_trend[SIG_RESONANCE]);
    printf("  novelty:   %.3f (raw: %.3f, trend: %+.3f)\n",
           ss->state.ema_signals[SIG_NOVELTY],
           ss->last_raw[SIG_NOVELTY],
           ss->state.signal_trend[SIG_NOVELTY]);
    printf("=========================\n");
}

void print_selfsense_stats(SelfSense* ss) {
    if (!ss->initialized) {
        printf("SelfSense: not initialized\n");
        return;
    }

    printf("\n=== SelfSense Stats ===\n");
    printf("Observations: %d\n", ss->observations);
    printf("Running loss: %.4f\n", ss->running_loss);
    printf("EMA momentum: %.2f\n", ss->state.ema_momentum);
    printf("History: %d entries\n", ss->state.history_count);
    printf("Identity: %s\n", ss->identity.initialized ? "initialized" : "not set");

    if (ss->state.history_count >= 16) {
        printf("Patterns:\n");
        printf("  Stuck:  %s\n", selfsense_detect_stuck(ss) ? "YES" : "no");
        printf("  Spiral: %s\n", selfsense_detect_spiral(ss) ? "YES" : "no");
    }
    printf("=======================\n");
}
