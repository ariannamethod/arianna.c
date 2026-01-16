/*
 * mathbrain.c - Arithmetic through resonance implementation
 *
 * "7 + 5 = 12 not because you memorized it,
 *  but because 7 and 5 resonate at a frequency that IS 12."
 *
 * Numbers have texture. Operations are attention patterns.
 * Learning happens through quality feedback, not backprop.
 */

#include "mathbrain.h"
#include <string.h>
#include <ctype.h>

// ============================================================
// Initialization
// ============================================================

// Xavier-ish initialization
static float rand_init(void) {
    return ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
}

void init_mathbrain(MathBrain* mb) {
    memset(mb, 0, sizeof(MathBrain));

    // Initialize number embeddings with structure
    for (int n = 0; n < MATH_MAX_NUM; n++) {
        // Each number gets a unique embedding
        for (int d = 0; d < MATH_DIM; d++) {
            float base = rand_init();

            // Add structural features
            // Magnitude: larger numbers have more "weight" in certain dims
            if (d < 8) {
                base += (float)n / MATH_MAX_NUM * 0.3f;
            }

            // Parity: even/odd signature
            if (d >= 8 && d < 16) {
                base += (n % 2 == 0) ? 0.2f : -0.2f;
            }

            // Divisibility patterns
            if (d >= 16 && d < 24) {
                if (n % 5 == 0) base += 0.15f;
                if (n % 10 == 0) base += 0.15f;
            }

            // Tens place
            if (d >= 24 && d < 32) {
                base += (float)(n / 10) / 10.0f * 0.2f;
            }

            mb->numbers.embeddings[n][d] = base;
        }
    }

    // Initialize magnitude and parity biases
    for (int d = 0; d < MATH_DIM; d++) {
        mb->numbers.magnitude_bias[d] = rand_init();
        mb->numbers.parity_bias[d] = rand_init();
    }

    // Initialize operation heads
    for (int op = 0; op < OP_COUNT; op++) {
        for (int i = 0; i < MATH_DIM; i++) {
            for (int j = 0; j < MATH_DIM; j++) {
                mb->ops.W[op][i][j] = rand_init();
            }
            mb->ops.b[op][i] = 0.0f;
        }

        // Different temperatures for different ops
        mb->ops.temperature[OP_ADD] = 0.3f;  // Addition is precise
        mb->ops.temperature[OP_SUB] = 0.3f;  // Subtraction is precise
        mb->ops.temperature[OP_MUL] = 0.5f;  // Multiplication needs more exploration
        mb->ops.temperature[OP_DIV] = 0.7f;  // Division is tricky (remainders, etc)
    }

    // Initialize MLP weights
    for (int i = 0; i < MATH_DIM * 2; i++) {
        for (int j = 0; j < MATH_HIDDEN; j++) {
            mb->hidden_w[i][j] = rand_init() / sqrtf((float)(MATH_DIM * 2));
        }
    }
    for (int j = 0; j < MATH_HIDDEN; j++) {
        mb->hidden_b[j] = 0.0f;
    }
    for (int i = 0; i < MATH_HIDDEN; i++) {
        for (int j = 0; j < MATH_DIM; j++) {
            mb->output_w[i][j] = rand_init() / sqrtf((float)MATH_HIDDEN);
        }
    }
    for (int j = 0; j < MATH_DIM; j++) {
        mb->output_b[j] = 0.0f;
    }

    // Learning params
    mb->lr = 0.01f;
    mb->momentum = 0.9f;

    // History
    mb->history.head = 0;
    mb->history.count = 0;
    mb->history.accuracy_ema = 0.5f;
    mb->history.confidence_ema = 0.5f;
    mb->history.total_computed = 0;
}

void free_mathbrain(MathBrain* mb) {
    // Nothing dynamic to free, but pattern for future
    (void)mb;
}

// ============================================================
// Number embeddings
// ============================================================

void get_number_embedding(MathBrain* mb, int n, float* out) {
    if (n < 0 || n >= MATH_MAX_NUM) {
        // Out of range: use modular arithmetic as fallback
        n = ((n % MATH_MAX_NUM) + MATH_MAX_NUM) % MATH_MAX_NUM;
    }

    memcpy(out, mb->numbers.embeddings[n], MATH_DIM * sizeof(float));
}

// Find closest number to an embedding
static int embedding_to_number(MathBrain* mb, float* embed, float* confidence) {
    int best = 0;
    float best_sim = -1e9f;
    float sum_exp = 0.0f;

    // Compute similarities with softmax
    float sims[MATH_MAX_NUM];
    float max_sim = -1e9f;

    for (int n = 0; n < MATH_MAX_NUM; n++) {
        float sim = 0.0f;
        float norm_e = 0.0f, norm_n = 0.0f;

        for (int d = 0; d < MATH_DIM; d++) {
            sim += embed[d] * mb->numbers.embeddings[n][d];
            norm_e += embed[d] * embed[d];
            norm_n += mb->numbers.embeddings[n][d] * mb->numbers.embeddings[n][d];
        }

        // Cosine similarity
        float denom = sqrtf(norm_e) * sqrtf(norm_n);
        if (denom > 1e-6f) {
            sim /= denom;
        }

        sims[n] = sim;
        if (sim > max_sim) {
            max_sim = sim;
            best = n;
            best_sim = sim;
        }
    }

    // Softmax for confidence
    for (int n = 0; n < MATH_MAX_NUM; n++) {
        sum_exp += expf((sims[n] - max_sim) * 5.0f);  // Temperature scaling
    }

    if (confidence) {
        *confidence = expf((best_sim - max_sim) * 5.0f) / sum_exp;
    }

    return best;
}

// ============================================================
// Operation computation
// ============================================================

// Apply operation-specific transformation
static void apply_op_transform(MathBrain* mb, MathOp op,
                                float* a_embed, float* b_embed,
                                float* out) {
    // Bilinear form: out = W @ (a ⊗ b) + b
    // Simplified: out[i] = sum_j(W[i][j] * a[j]) * sum_k(W[i][k] * b[k])

    float a_proj[MATH_DIM], b_proj[MATH_DIM];

    // Project both embeddings
    for (int i = 0; i < MATH_DIM; i++) {
        a_proj[i] = 0.0f;
        b_proj[i] = 0.0f;
        for (int j = 0; j < MATH_DIM; j++) {
            a_proj[i] += mb->ops.W[op][i][j] * a_embed[j];
            b_proj[i] += mb->ops.W[op][i][j] * b_embed[j];
        }
    }

    // Combine based on operation
    for (int i = 0; i < MATH_DIM; i++) {
        switch (op) {
            case OP_ADD:
                // Addition: additive combination
                out[i] = a_proj[i] + b_proj[i] + mb->ops.b[op][i];
                break;
            case OP_SUB:
                // Subtraction: difference
                out[i] = a_proj[i] - b_proj[i] + mb->ops.b[op][i];
                break;
            case OP_MUL:
                // Multiplication: multiplicative (element-wise)
                out[i] = a_proj[i] * b_proj[i] + mb->ops.b[op][i];
                break;
            case OP_DIV:
                // Division: ratio (with safeguard)
                if (fabsf(b_proj[i]) > 0.01f) {
                    out[i] = a_proj[i] / b_proj[i] + mb->ops.b[op][i];
                } else {
                    out[i] = a_proj[i] * 10.0f + mb->ops.b[op][i];  // Large number signal
                }
                break;
            default:
                out[i] = a_proj[i] + b_proj[i];
        }
    }
}

int compute_op(MathBrain* mb, int a, MathOp op, int b, float* confidence) {
    float a_embed[MATH_DIM], b_embed[MATH_DIM];
    float combined[MATH_DIM * 2];
    float hidden[MATH_HIDDEN];
    float output[MATH_DIM];

    // Get embeddings
    get_number_embedding(mb, a, a_embed);
    get_number_embedding(mb, b, b_embed);

    // Save for learning
    memcpy(mb->last_a_embed, a_embed, MATH_DIM * sizeof(float));
    memcpy(mb->last_b_embed, b_embed, MATH_DIM * sizeof(float));

    // Apply operation transform
    float op_out[MATH_DIM];
    apply_op_transform(mb, op, a_embed, b_embed, op_out);

    // Concatenate a, b embeddings for MLP
    memcpy(combined, a_embed, MATH_DIM * sizeof(float));
    memcpy(combined + MATH_DIM, b_embed, MATH_DIM * sizeof(float));

    // MLP forward: hidden = ReLU(W1 @ concat + b1)
    for (int j = 0; j < MATH_HIDDEN; j++) {
        float sum = mb->hidden_b[j];
        for (int i = 0; i < MATH_DIM * 2; i++) {
            sum += combined[i] * mb->hidden_w[i][j];
        }
        hidden[j] = fmaxf(0.0f, sum);  // ReLU
    }
    memcpy(mb->last_hidden, hidden, MATH_HIDDEN * sizeof(float));

    // Output: result_embed = W2 @ hidden + b2
    for (int j = 0; j < MATH_DIM; j++) {
        float sum = mb->output_b[j];
        for (int i = 0; i < MATH_HIDDEN; i++) {
            sum += hidden[i] * mb->output_w[i][j];
        }
        output[j] = sum;
    }

    // Blend MLP output with operation-specific transform
    for (int i = 0; i < MATH_DIM; i++) {
        output[i] = 0.6f * output[i] + 0.4f * op_out[i];
    }
    memcpy(mb->last_output, output, MATH_DIM * sizeof(float));

    // Resonance boost: if we've seen this pair before, bias toward common result
    if (mb->resonance.total_observations > 10 && a < MATH_MAX_NUM && b < MATH_MAX_NUM) {
        float pair_weight = mb->resonance.pair_freq[a][b];
        if (pair_weight > 0.0f) {
            // Find most common result for this pair
            int best_result = 0;
            float best_assoc = 0.0f;
            for (int r = 0; r < MATH_MAX_NUM; r++) {
                float assoc = mb->resonance.result_assoc[a][b][r];
                if (assoc > best_assoc) {
                    best_assoc = assoc;
                    best_result = r;
                }
            }

            // Blend in the common result's embedding
            if (best_assoc > 0.1f) {
                float result_embed[MATH_DIM];
                get_number_embedding(mb, best_result, result_embed);
                float blend = fminf(0.3f, best_assoc);
                for (int i = 0; i < MATH_DIM; i++) {
                    output[i] = (1.0f - blend) * output[i] + blend * result_embed[i];
                }
            }
        }
    }

    // Decode: find closest number embedding
    int result = embedding_to_number(mb, output, confidence);
    mb->last_predicted = result;

    return result;
}

// ============================================================
// Learning
// ============================================================

void mathbrain_learn(MathBrain* mb, int a, MathOp op, int b, int correct_result) {
    // Get correct embedding
    float correct_embed[MATH_DIM];
    get_number_embedding(mb, correct_result, correct_embed);

    // Compute error
    float error[MATH_DIM];
    float error_mag = 0.0f;
    for (int i = 0; i < MATH_DIM; i++) {
        error[i] = correct_embed[i] - mb->last_output[i];
        error_mag += error[i] * error[i];
    }
    error_mag = sqrtf(error_mag);

    // Was prediction correct?
    int was_correct = (mb->last_predicted == correct_result);

    // Record in history
    MathEvent* evt = &mb->history.history[mb->history.head];
    evt->a = a;
    evt->b = b;
    evt->op = op;
    evt->result = mb->last_predicted;
    evt->correct = was_correct;
    evt->confidence = 1.0f / (1.0f + error_mag);  // Higher confidence when closer

    mb->history.head = (mb->history.head + 1) % MATH_HISTORY;
    if (mb->history.count < MATH_HISTORY) mb->history.count++;
    mb->history.total_computed++;

    // Update accuracy EMA
    mb->history.accuracy_ema = 0.95f * mb->history.accuracy_ema +
                               0.05f * (was_correct ? 1.0f : 0.0f);

    // Skip learning if correct (or learn less)
    float learn_scale = was_correct ? 0.1f : 1.0f;

    // Update output weights: W2 += lr * hidden @ error^T
    for (int i = 0; i < MATH_HIDDEN; i++) {
        for (int j = 0; j < MATH_DIM; j++) {
            mb->output_w[i][j] += mb->lr * learn_scale *
                                  mb->last_hidden[i] * error[j];
        }
    }
    for (int j = 0; j < MATH_DIM; j++) {
        mb->output_b[j] += mb->lr * learn_scale * error[j];
    }

    // Backprop through ReLU to hidden: d_hidden = (hidden > 0) * W2^T @ error
    float d_hidden[MATH_HIDDEN];
    for (int i = 0; i < MATH_HIDDEN; i++) {
        d_hidden[i] = 0.0f;
        if (mb->last_hidden[i] > 0.0f) {
            for (int j = 0; j < MATH_DIM; j++) {
                d_hidden[i] += mb->output_w[i][j] * error[j];
            }
        }
    }

    // Update hidden weights: W1 += lr * concat @ d_hidden^T
    float combined[MATH_DIM * 2];
    memcpy(combined, mb->last_a_embed, MATH_DIM * sizeof(float));
    memcpy(combined + MATH_DIM, mb->last_b_embed, MATH_DIM * sizeof(float));

    for (int i = 0; i < MATH_DIM * 2; i++) {
        for (int j = 0; j < MATH_HIDDEN; j++) {
            mb->hidden_w[i][j] += mb->lr * learn_scale * 0.1f *
                                  combined[i] * d_hidden[j];
        }
    }

    // Update operation weights (slower learning for structural weights)
    float op_lr = mb->lr * 0.01f * learn_scale;
    for (int i = 0; i < MATH_DIM; i++) {
        for (int j = 0; j < MATH_DIM; j++) {
            // Gradient approximation
            float grad = error[i] * (mb->last_a_embed[j] + mb->last_b_embed[j]);
            mb->ops.W[op][i][j] += op_lr * grad;
        }
        mb->ops.b[op][i] += op_lr * error[i];
    }

    // Update resonance field
    observe_math(mb, a, b, correct_result);
}

void observe_math(MathBrain* mb, int a, int b, int result) {
    if (a >= MATH_MAX_NUM || b >= MATH_MAX_NUM || result >= MATH_MAX_NUM) return;
    if (a < 0 || b < 0 || result < 0) return;

    // Update pair frequency
    mb->resonance.pair_freq[a][b] += 1.0f;

    // Update result association
    mb->resonance.result_assoc[a][b][result] += 1.0f;

    mb->resonance.total_observations++;

    // Decay old observations to prevent saturation
    if (mb->resonance.total_observations % 1000 == 0) {
        for (int i = 0; i < MATH_MAX_NUM; i++) {
            for (int j = 0; j < MATH_MAX_NUM; j++) {
                mb->resonance.pair_freq[i][j] *= 0.99f;
                for (int k = 0; k < MATH_MAX_NUM; k++) {
                    mb->resonance.result_assoc[i][j][k] *= 0.99f;
                }
            }
        }
    }
}

// ============================================================
// Text parsing
// ============================================================

int parse_math_expr(const char* text, int* a, MathOp* op, int* b) {
    // Parse "7 + 5" or "12*3" etc.
    const char* p = text;

    // Skip whitespace
    while (*p && isspace(*p)) p++;

    // Parse first number
    if (!isdigit(*p)) return 0;
    *a = 0;
    while (isdigit(*p)) {
        *a = *a * 10 + (*p - '0');
        p++;
    }

    // Skip whitespace
    while (*p && isspace(*p)) p++;

    // Parse operator
    switch (*p) {
        case '+': *op = OP_ADD; break;
        case '-': *op = OP_SUB; break;
        case '*': case 'x': case 'X': *op = OP_MUL; break;
        case '/': *op = OP_DIV; break;
        default: return 0;
    }
    p++;

    // Skip whitespace
    while (*p && isspace(*p)) p++;

    // Parse second number
    if (!isdigit(*p)) return 0;
    *b = 0;
    while (isdigit(*p)) {
        *b = *b * 10 + (*p - '0');
        p++;
    }

    return 1;
}

void result_to_text(int result, char* out, int max_len) {
    snprintf(out, max_len, "%d", result);
}

int compute_from_text(MathBrain* mb, const char* expr, char* result, int max_len) {
    int a, b;
    MathOp op;

    if (!parse_math_expr(expr, &a, &op, &b)) {
        snprintf(result, max_len, "[parse error]");
        return -1;
    }

    float confidence;
    int predicted = compute_op(mb, a, op, b, &confidence);

    // Compute ground truth for learning
    int truth;
    switch (op) {
        case OP_ADD: truth = a + b; break;
        case OP_SUB: truth = a - b; break;
        case OP_MUL: truth = a * b; break;
        case OP_DIV: truth = (b != 0) ? a / b : 0; break;
        default: truth = 0;
    }

    // Learn from this computation
    if (truth >= 0 && truth < MATH_MAX_NUM) {
        mathbrain_learn(mb, a, op, b, truth);
    }

    // Return prediction (not truth - we want to see what the model thinks)
    result_to_text(predicted, result, max_len);

    return (predicted == truth) ? 1 : 0;
}

// ============================================================
// Stats and debugging
// ============================================================

void print_mathbrain_stats(MathBrain* mb) {
    printf("\n=== MathBrain Stats ===\n");
    printf("Total computations: %d\n", mb->history.total_computed);
    printf("Accuracy (EMA):     %.1f%%\n", mb->history.accuracy_ema * 100.0f);
    printf("Confidence (EMA):   %.2f\n", mb->history.confidence_ema);
    printf("Resonance obs:      %d\n", mb->resonance.total_observations);

    // Recent history
    if (mb->history.count > 0) {
        printf("Recent:\n");
        int show = (mb->history.count < 5) ? mb->history.count : 5;
        int start = (mb->history.head - show + MATH_HISTORY) % MATH_HISTORY;
        for (int i = 0; i < show; i++) {
            int idx = (start + i) % MATH_HISTORY;
            MathEvent* evt = &mb->history.history[idx];
            const char* ops[] = {"+", "-", "*", "/"};
            printf("  %d %s %d = %d %s\n",
                   evt->a, ops[evt->op], evt->b, evt->result,
                   evt->correct ? "[correct]" : "[wrong]");
        }
    }
    printf("=======================\n");
}

float get_recent_accuracy(MathBrain* mb, int window) {
    if (mb->history.count == 0) return 0.5f;

    int count = 0;
    int correct = 0;
    int n = (window < mb->history.count) ? window : mb->history.count;

    for (int i = 0; i < n; i++) {
        int idx = (mb->history.head - 1 - i + MATH_HISTORY) % MATH_HISTORY;
        if (mb->history.history[idx].correct) correct++;
        count++;
    }

    return (float)correct / (float)count;
}

void print_number_similarities(MathBrain* mb) {
    printf("\nNumber embedding similarities (0-9):\n");
    printf("    ");
    for (int j = 0; j < 10; j++) printf(" %3d", j);
    printf("\n");

    for (int i = 0; i < 10; i++) {
        printf("%3d:", i);
        for (int j = 0; j < 10; j++) {
            float sim = 0.0f;
            float norm_i = 0.0f, norm_j = 0.0f;
            for (int d = 0; d < MATH_DIM; d++) {
                sim += mb->numbers.embeddings[i][d] * mb->numbers.embeddings[j][d];
                norm_i += mb->numbers.embeddings[i][d] * mb->numbers.embeddings[i][d];
                norm_j += mb->numbers.embeddings[j][d] * mb->numbers.embeddings[j][d];
            }
            sim /= sqrtf(norm_i) * sqrtf(norm_j);
            printf(" %3.0f", sim * 100);
        }
        printf("\n");
    }
}

// ============================================================
// Save/Load
// ============================================================

int save_mathbrain(MathBrain* mb, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Magic header
    const char magic[] = "MATHBRAIN1";
    fwrite(magic, 1, 10, f);

    // Number embeddings
    fwrite(mb->numbers.embeddings, sizeof(float), MATH_MAX_NUM * MATH_DIM, f);
    fwrite(mb->numbers.magnitude_bias, sizeof(float), MATH_DIM, f);
    fwrite(mb->numbers.parity_bias, sizeof(float), MATH_DIM, f);

    // Operation weights
    fwrite(mb->ops.W, sizeof(float), OP_COUNT * MATH_DIM * MATH_DIM, f);
    fwrite(mb->ops.b, sizeof(float), OP_COUNT * MATH_DIM, f);
    fwrite(mb->ops.temperature, sizeof(float), OP_COUNT, f);

    // MLP weights
    fwrite(mb->hidden_w, sizeof(float), MATH_DIM * 2 * MATH_HIDDEN, f);
    fwrite(mb->hidden_b, sizeof(float), MATH_HIDDEN, f);
    fwrite(mb->output_w, sizeof(float), MATH_HIDDEN * MATH_DIM, f);
    fwrite(mb->output_b, sizeof(float), MATH_DIM, f);

    // Stats
    fwrite(&mb->history.accuracy_ema, sizeof(float), 1, f);
    fwrite(&mb->history.total_computed, sizeof(int), 1, f);

    fclose(f);
    return 0;
}

int load_mathbrain(MathBrain* mb, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    // Check magic
    char magic[10];
    if (fread(magic, 1, 10, f) != 10 || strncmp(magic, "MATHBRAIN1", 10) != 0) {
        fclose(f);
        return -1;
    }

    // Number embeddings
    if (fread(mb->numbers.embeddings, sizeof(float), MATH_MAX_NUM * MATH_DIM, f) !=
        MATH_MAX_NUM * MATH_DIM) {
        fclose(f);
        return -1;
    }
    fread(mb->numbers.magnitude_bias, sizeof(float), MATH_DIM, f);
    fread(mb->numbers.parity_bias, sizeof(float), MATH_DIM, f);

    // Operation weights
    fread(mb->ops.W, sizeof(float), OP_COUNT * MATH_DIM * MATH_DIM, f);
    fread(mb->ops.b, sizeof(float), OP_COUNT * MATH_DIM, f);
    fread(mb->ops.temperature, sizeof(float), OP_COUNT, f);

    // MLP weights
    fread(mb->hidden_w, sizeof(float), MATH_DIM * 2 * MATH_HIDDEN, f);
    fread(mb->hidden_b, sizeof(float), MATH_HIDDEN, f);
    fread(mb->output_w, sizeof(float), MATH_HIDDEN * MATH_DIM, f);
    fread(mb->output_b, sizeof(float), MATH_DIM, f);

    // Stats
    fread(&mb->history.accuracy_ema, sizeof(float), 1, f);
    fread(&mb->history.total_computed, sizeof(int), 1, f);

    fclose(f);
    return 0;
}
