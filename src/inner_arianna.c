// inner_arianna.c — MetaVoice: Inner Arianna (борьба)
// "Two voices, one Arianna"
//
// Main Arianna (base) vs Inner Arianna (dialogue LoRA)
// Борьба determines final output
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "inner_arianna.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT / FREE
// ═══════════════════════════════════════════════════════════════════════════════

void inner_init(InnerArianna* ia) {
    memset(ia, 0, sizeof(InnerArianna));
    ia->borba_mode = BORBA_MODE_BLEND;
    ia->inner_weight = 0.3f;  // default: 30% inner voice
    ia->lora_rank = 0;
}

void inner_free(InnerArianna* ia) {
    free(ia->lora_A);
    free(ia->lora_B);
    memset(ia, 0, sizeof(InnerArianna));
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOAD DIALOGUE LORA
// ═══════════════════════════════════════════════════════════════════════════════

int inner_load_lora(InnerArianna* ia, const char* lora_path) {
    FILE* f = fopen(lora_path, "rb");
    if (!f) {
        fprintf(stderr, "[inner] cannot open LoRA: %s\n", lora_path);
        return -1;
    }

    // LoRA header: rank, layers, dim
    int header[3];
    if (fread(header, sizeof(int), 3, f) != 3) {
        fclose(f);
        return -1;
    }

    ia->lora_rank = header[0];
    ia->lora_layers = header[1];
    int dim = header[2];

    // Allocate LoRA matrices
    size_t A_size = ia->lora_layers * dim * ia->lora_rank;
    size_t B_size = ia->lora_layers * ia->lora_rank * dim;

    ia->lora_A = (float*)malloc(A_size * sizeof(float));
    ia->lora_B = (float*)malloc(B_size * sizeof(float));

    if (!ia->lora_A || !ia->lora_B) {
        fclose(f);
        return -1;
    }

    // Read weights
    fread(ia->lora_A, sizeof(float), A_size, f);
    fread(ia->lora_B, sizeof(float), B_size, f);

    fclose(f);

    fprintf(stderr, "[inner] loaded LoRA: rank=%d layers=%d dim=%d\n",
            ia->lora_rank, ia->lora_layers, dim);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MATH HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

static void softmax_inplace(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) {
            x[i] /= sum;
        }
    }
}

float inner_compute_entropy(const float* logits, int vocab_size) {
    // Compute entropy of probability distribution
    // First apply softmax
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    memcpy(probs, logits, vocab_size * sizeof(float));
    softmax_inplace(probs, vocab_size);

    float entropy = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (probs[i] > 1e-10f) {
            entropy -= probs[i] * logf(probs[i]);
        }
    }

    free(probs);
    return entropy;
}

float inner_compute_divergence(const float* main_logits,
                               const float* inner_logits,
                               int vocab_size) {
    // KL divergence from main to inner
    float* main_probs = (float*)malloc(vocab_size * sizeof(float));
    float* inner_probs = (float*)malloc(vocab_size * sizeof(float));

    memcpy(main_probs, main_logits, vocab_size * sizeof(float));
    memcpy(inner_probs, inner_logits, vocab_size * sizeof(float));

    softmax_inplace(main_probs, vocab_size);
    softmax_inplace(inner_probs, vocab_size);

    float kl = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (main_probs[i] > 1e-10f && inner_probs[i] > 1e-10f) {
            kl += main_probs[i] * logf(main_probs[i] / inner_probs[i]);
        }
    }

    free(main_probs);
    free(inner_probs);

    return kl;
}

// ═══════════════════════════════════════════════════════════════════════════════
// БОРЬБА — the struggle between voices
// ═══════════════════════════════════════════════════════════════════════════════

void inner_full_borba(InnerArianna* ia,
                      float* output_logits,
                      const float* main_logits,
                      const float* inner_logits,
                      int vocab_size) {
    // Compute statistics
    ia->last_main_entropy = inner_compute_entropy(main_logits, vocab_size);
    ia->last_inner_entropy = inner_compute_entropy(inner_logits, vocab_size);
    ia->last_divergence = inner_compute_divergence(main_logits, inner_logits, vocab_size);

    // Running average of divergence
    ia->avg_divergence = 0.9f * ia->avg_divergence + 0.1f * ia->last_divergence;

    switch (ia->borba_mode) {
        case BORBA_MODE_BLEND: {
            // Weighted blend of logits
            float w = ia->inner_weight;
            for (int i = 0; i < vocab_size; i++) {
                output_logits[i] = (1.0f - w) * main_logits[i] + w * inner_logits[i];
            }
            // No clear winner in blend mode
            ia->last_winner = -1;
            break;
        }

        case BORBA_MODE_ENTROPY: {
            // Choose voice with lower entropy (more confident)
            if (ia->last_inner_entropy < ia->last_main_entropy) {
                memcpy(output_logits, inner_logits, vocab_size * sizeof(float));
                ia->last_winner = 1;
                ia->inner_wins++;
            } else {
                memcpy(output_logits, main_logits, vocab_size * sizeof(float));
                ia->last_winner = 0;
                ia->main_wins++;
            }
            break;
        }

        case BORBA_MODE_COHERENCE: {
            // In coherence mode, prefer main (more stable)
            // but switch to inner if divergence is low (they agree)
            if (ia->last_divergence < 0.1f) {
                // Voices agree - use inner (more expressive)
                memcpy(output_logits, inner_logits, vocab_size * sizeof(float));
                ia->last_winner = 1;
                ia->inner_wins++;
            } else {
                // Voices disagree - use main (more stable)
                memcpy(output_logits, main_logits, vocab_size * sizeof(float));
                ia->last_winner = 0;
                ia->main_wins++;
            }
            break;
        }

        case BORBA_MODE_SURPRISE: {
            // Choose voice with higher entropy (more surprising)
            if (ia->last_inner_entropy > ia->last_main_entropy) {
                memcpy(output_logits, inner_logits, vocab_size * sizeof(float));
                ia->last_winner = 1;
                ia->inner_wins++;
            } else {
                memcpy(output_logits, main_logits, vocab_size * sizeof(float));
                ia->last_winner = 0;
                ia->main_wins++;
            }
            break;
        }

        case BORBA_MODE_RANDOM: {
            // Random choice per token
            float r = (float)rand() / (float)RAND_MAX;
            if (r < ia->inner_weight) {
                memcpy(output_logits, inner_logits, vocab_size * sizeof(float));
                ia->last_winner = 1;
                ia->inner_wins++;
            } else {
                memcpy(output_logits, main_logits, vocab_size * sizeof(float));
                ia->last_winner = 0;
                ia->main_wins++;
            }
            break;
        }

        default:
            // Fallback to main
            memcpy(output_logits, main_logits, vocab_size * sizeof(float));
            ia->last_winner = 0;
    }
}

void inner_apply_borba(InnerArianna* ia,
                       float* main_logits,
                       const float* inner_logits,
                       int vocab_size) {
    // In-place version - modifies main_logits
    float* temp = (float*)malloc(vocab_size * sizeof(float));
    inner_full_borba(ia, temp, main_logits, inner_logits, vocab_size);
    memcpy(main_logits, temp, vocab_size * sizeof(float));
    free(temp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LORA DELTA — apply inner voice modification to hidden state
// ═══════════════════════════════════════════════════════════════════════════════

void inner_apply_lora_delta(InnerArianna* ia,
                            float* hidden_state,
                            int layer,
                            int dim) {
    if (!ia->lora_A || !ia->lora_B || layer >= ia->lora_layers) {
        return;
    }

    int rank = ia->lora_rank;

    // LoRA delta: h' = h + B @ A @ h
    // Step 1: intermediate = A @ h (dim → rank)
    float* intermediate = (float*)malloc(rank * sizeof(float));
    float* A = ia->lora_A + layer * dim * rank;

    for (int r = 0; r < rank; r++) {
        intermediate[r] = 0.0f;
        for (int d = 0; d < dim; d++) {
            intermediate[r] += A[d * rank + r] * hidden_state[d];
        }
    }

    // Step 2: delta = B @ intermediate (rank → dim)
    float* B = ia->lora_B + layer * rank * dim;

    for (int d = 0; d < dim; d++) {
        float delta = 0.0f;
        for (int r = 0; r < rank; r++) {
            delta += B[r * dim + d] * intermediate[r];
        }
        // Apply with scaling
        hidden_state[d] += delta * ia->inner_weight;
    }

    free(intermediate);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETTERS
// ═══════════════════════════════════════════════════════════════════════════════

void inner_set_mode(InnerArianna* ia, int mode) {
    if (mode >= 0 && mode <= BORBA_MODE_RANDOM) {
        ia->borba_mode = mode;
    }
}

void inner_set_weight(InnerArianna* ia, float weight) {
    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;
    ia->inner_weight = weight;
}
