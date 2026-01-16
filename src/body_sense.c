/*
 * body_sense.c - Somatic Awareness for Arianna
 *
 * "I do not read words, I read numbers." - Stanley
 *
 * Implementation of:
 * - BodyState: entropy, novelty, arousal, valence + metrics
 * - Regulation scores: boredom, overwhelm, stuck
 * - Tiny MLP for quality prediction
 * - Temperature/expert regulation based on somatic state
 */

#include "body_sense.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================
// Random number generation
// ============================================================

static unsigned int body_rng_state = 42;

static float body_randf(void) {
    body_rng_state = body_rng_state * 1103515245 + 12345;
    return (float)(body_rng_state % 10000) / 10000.0f;
}

static float body_randn(void) {
    // Box-Muller transform for Gaussian
    float u1 = body_randf();
    float u2 = body_randf();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

// ============================================================
// Activation functions
// ============================================================

static float tanhf_approx(float x) {
    // Fast tanh approximation
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// ============================================================
// MLP Implementation
// ============================================================

void init_body_mlp(BodyMLP* mlp) {
    // Xavier initialization
    float scale1 = sqrtf(2.0f / BODY_INPUT_DIM);
    float scale2 = sqrtf(2.0f / BODY_HIDDEN_DIM);

    // Layer 1
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        for (int j = 0; j < BODY_INPUT_DIM; j++) {
            mlp->w1[i][j] = body_randn() * scale1;
        }
        mlp->b1[i] = 0.0f;
    }

    // Layer 2
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        mlp->w2[i] = body_randn() * scale2;
    }
    mlp->b2 = 0.0f;

    // Clear hidden
    memset(mlp->hidden, 0, sizeof(mlp->hidden));
}

void state_to_features(BodyState* state, float* features) {
    // Normalize all inputs to [0,1] range
    features[0] = state->entropy;
    features[1] = state->novelty;
    features[2] = state->arousal;
    features[3] = (state->valence + 1.0f) / 2.0f;  // [-1,1] → [0,1]
    features[4] = clampf((float)state->total_shards / 100.0f, 0.0f, 1.0f);
    features[5] = clampf((float)state->active_shards / 20.0f, 0.0f, 1.0f);
    features[6] = state->avg_resonance;
    features[7] = clampf((float)state->overthinking_depth / 5.0f, 0.0f, 1.0f);
    features[8] = clampf((float)state->meta_patterns / 50.0f, 0.0f, 1.0f);
    features[9] = state->expert_temp;
    features[10] = clampf((float)state->response_length / 64.0f, 0.0f, 1.0f);
    features[11] = state->unique_ratio;
    features[12] = state->quality;
    features[13] = (float)(state->expert_id) / 3.0f;  // 0-3 → 0-1
}

float mlp_predict(BodyMLP* mlp, float* features) {
    // Layer 1: input → hidden (with tanh)
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < BODY_INPUT_DIM; j++) {
            sum += mlp->w1[i][j] * features[j];
        }
        mlp->hidden[i] = tanhf_approx(sum);
    }

    // Layer 2: hidden → output
    float output = mlp->b2;
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        output += mlp->w2[i] * mlp->hidden[i];
    }

    // Sigmoid for [0,1] output
    output = 1.0f / (1.0f + expf(-output));

    return clampf(output, 0.0f, 1.0f);
}

float mlp_learn(BodyMLP* mlp, float* features, float target,
               float learning_rate) {
    // Forward pass
    float pred = mlp_predict(mlp, features);

    // Compute loss (MSE)
    float error = pred - target;
    float loss = error * error;

    // Backprop (simplified gradient descent)
    // d_loss/d_pred = 2 * error
    // d_pred/d_output = pred * (1 - pred)  (sigmoid derivative)
    float d_output = 2.0f * error * pred * (1.0f - pred);

    // Layer 2 gradients
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        float grad = d_output * mlp->hidden[i];
        mlp->w2[i] -= learning_rate * grad;
        mlp->w2[i] = clampf(mlp->w2[i], -5.0f, 5.0f);
    }
    mlp->b2 -= learning_rate * d_output;
    mlp->b2 = clampf(mlp->b2, -5.0f, 5.0f);

    // Layer 1 gradients
    for (int i = 0; i < BODY_HIDDEN_DIM; i++) {
        float d_hidden = d_output * mlp->w2[i];
        // tanh derivative: 1 - tanh^2
        float tanh_deriv = 1.0f - mlp->hidden[i] * mlp->hidden[i];
        float d_pre = d_hidden * tanh_deriv;

        for (int j = 0; j < BODY_INPUT_DIM; j++) {
            float grad = d_pre * features[j];
            mlp->w1[i][j] -= learning_rate * grad;
            mlp->w1[i][j] = clampf(mlp->w1[i][j], -5.0f, 5.0f);
        }
        mlp->b1[i] -= learning_rate * d_pre;
        mlp->b1[i] = clampf(mlp->b1[i], -5.0f, 5.0f);
    }

    return loss;
}

// ============================================================
// Body State Operations
// ============================================================

void init_body_state(BodyState* state) {
    state->entropy = 0.5f;
    state->novelty = 0.5f;
    state->arousal = 0.5f;
    state->valence = 0.0f;

    state->active_shards = 0;
    state->total_shards = 0;
    state->avg_resonance = 0.5f;

    state->overthinking_depth = 0;
    state->meta_patterns = 0;

    state->response_length = 0;
    state->unique_ratio = 0.5f;

    state->quality = 0.5f;

    state->expert_id = 0;
    state->expert_temp = 0.8f;
}

void update_body_state(BodyState* state,
                      float entropy, float novelty,
                      float arousal, float valence,
                      int response_length, float unique_ratio) {
    state->entropy = clampf(entropy, 0.0f, 1.0f);
    state->novelty = clampf(novelty, 0.0f, 1.0f);
    state->arousal = clampf(arousal, 0.0f, 1.0f);
    state->valence = clampf(valence, -1.0f, 1.0f);
    state->response_length = response_length;
    state->unique_ratio = clampf(unique_ratio, 0.0f, 1.0f);
}

void set_quality_feedback(BodyState* state, float quality) {
    state->quality = clampf(quality, 0.0f, 1.0f);
}

// ============================================================
// Regulation Scores
// ============================================================

float compute_boredom(BodyState* state) {
    /*
     * Boredom = low novelty + low arousal + low entropy + shallow depth
     * "When everything feels flat and predictable"
     */
    float novelty_component = fmaxf(0.0f, 1.0f - state->novelty);
    float arousal_component = fmaxf(0.0f, 1.0f - state->arousal);
    float entropy_component = fmaxf(0.0f, 1.0f - state->entropy);
    float depth_component = fmaxf(0.0f, 1.0f - state->overthinking_depth / 5.0f);

    float score = (
        0.35f * novelty_component +
        0.30f * arousal_component +
        0.20f * entropy_component +
        0.15f * depth_component
    );

    return clampf(score, 0.0f, 1.0f);
}

float compute_overwhelm(BodyState* state) {
    /*
     * Overwhelm = high arousal + high entropy + negative valence
     * "When there's too much happening, too intensely"
     */
    float arousal_component = state->arousal;
    float entropy_component = state->entropy;
    float valence_component = fmaxf(0.0f, -state->valence);

    float score = (
        0.45f * arousal_component +
        0.35f * entropy_component +
        0.20f * valence_component
    );

    return clampf(score, 0.0f, 1.0f);
}

float compute_stuck(BodyState* state, float predicted_quality) {
    /*
     * Stuck = low quality + repetition + no crystallization
     * "When we're going in circles without progress"
     */
    float quality_component = fmaxf(0.0f, 1.0f - predicted_quality);
    float repetition_component = fmaxf(0.0f, 1.0f - state->unique_ratio);
    // No crystallization = worse
    float crystal_component = (state->meta_patterns > 0) ? 0.0f : 0.5f;

    float score = (
        0.50f * quality_component +
        0.30f * repetition_component +
        0.20f * crystal_component
    );

    return clampf(score, 0.0f, 1.0f);
}

// ============================================================
// Body Sense Main Functions
// ============================================================

void init_body_sense(BodySense* bs) {
    body_rng_state = (unsigned int)time(NULL);

    init_body_mlp(&bs->mlp);
    init_body_state(&bs->current_state);

    bs->learning_rate = 0.01f;
    bs->observations = 0;
    bs->running_loss = 0.0f;
    bs->last_loss = 0.0f;

    bs->history_idx = 0;
    bs->history_count = 0;

    bs->initialized = 1;
}

void free_body_sense(BodySense* bs) {
    bs->initialized = 0;
}

float body_observe(BodySense* bs, BodyState* state) {
    if (!bs->initialized) return 0.0f;

    // Convert state to features
    float features[BODY_INPUT_DIM];
    state_to_features(state, features);

    // Learn from quality feedback
    float loss = mlp_learn(&bs->mlp, features, state->quality, bs->learning_rate);

    // Update running stats
    bs->observations++;
    bs->last_loss = loss;
    bs->running_loss += (loss - bs->running_loss) * 0.05f;

    return loss;
}

float body_predict(BodySense* bs, BodyState* state) {
    if (!bs->initialized) return 0.5f;

    float features[BODY_INPUT_DIM];
    state_to_features(state, features);

    return mlp_predict(&bs->mlp, features);
}

RegulationResult body_regulate(BodySense* bs, BodyState* state,
                               float current_temp, int current_expert) {
    RegulationResult result;

    // Predict quality
    float predicted_q = body_predict(bs, state);

    // Compute regulation scores
    float boredom = compute_boredom(state);
    float overwhelm = compute_overwhelm(state);
    float stuck = compute_stuck(state, predicted_q);

    result.boredom = boredom;
    result.overwhelm = overwhelm;
    result.stuck = stuck;
    result.predicted_quality = predicted_q;

    // Adjust temperature based on somatic state
    float temp_nudge = 0.0f;
    int suggested_expert = current_expert;

    // Boredom → increase temperature, switch to creative
    if (boredom > 0.6f) {
        temp_nudge += TEMP_NUDGE_MAX * (boredom - 0.6f) / 0.4f;
        if (boredom > 0.75f && current_expert != 2) {  // Not creative
            suggested_expert = 2;  // Creative
        }
    }

    // Overwhelm → decrease temperature, switch to precise/structural
    if (overwhelm > 0.7f) {
        temp_nudge -= TEMP_NUDGE_MAX * (overwhelm - 0.7f) / 0.3f;
        if (overwhelm > 0.85f && current_expert != 3 && current_expert != 0) {
            suggested_expert = 3;  // Precise
        }
    }

    // Stuck → small temp increase, switch to semantic
    if (stuck > 0.6f) {
        temp_nudge += 0.1f;
        if (stuck > 0.75f && current_expert == 0) {  // Structural
            suggested_expert = 1;  // Semantic
        }
    }

    // Apply temperature adjustment
    result.temperature = clampf(current_temp + temp_nudge, TEMP_MIN, TEMP_MAX);
    result.expert_id = suggested_expert;

    // Store in history
    bs->history[bs->history_idx] = result;
    bs->history_idx = (bs->history_idx + 1) % BODY_HISTORY_SIZE;
    if (bs->history_count < BODY_HISTORY_SIZE) {
        bs->history_count++;
    }

    return result;
}

void body_get_stats(BodySense* bs, float* avg_boredom,
                   float* avg_overwhelm, float* avg_stuck) {
    *avg_boredom = 0.0f;
    *avg_overwhelm = 0.0f;
    *avg_stuck = 0.0f;

    if (bs->history_count == 0) return;

    for (int i = 0; i < bs->history_count; i++) {
        *avg_boredom += bs->history[i].boredom;
        *avg_overwhelm += bs->history[i].overwhelm;
        *avg_stuck += bs->history[i].stuck;
    }

    *avg_boredom /= bs->history_count;
    *avg_overwhelm /= bs->history_count;
    *avg_stuck /= bs->history_count;
}

// ============================================================
// File I/O
// ============================================================

int save_body_sense(BodySense* bs, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    const char magic[] = "BODY";
    fwrite(magic, 1, 4, f);

    fwrite(&bs->observations, sizeof(int), 1, f);
    fwrite(&bs->running_loss, sizeof(float), 1, f);
    fwrite(&bs->learning_rate, sizeof(float), 1, f);

    // Write MLP weights
    fwrite(bs->mlp.w1, sizeof(float), BODY_HIDDEN_DIM * BODY_INPUT_DIM, f);
    fwrite(bs->mlp.b1, sizeof(float), BODY_HIDDEN_DIM, f);
    fwrite(bs->mlp.w2, sizeof(float), BODY_HIDDEN_DIM, f);
    fwrite(&bs->mlp.b2, sizeof(float), 1, f);

    fclose(f);
    return 1;
}

int load_body_sense(BodySense* bs, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BODY", 4) != 0) {
        fclose(f);
        return 0;
    }

    if (fread(&bs->observations, sizeof(int), 1, f) != 1 ||
        fread(&bs->running_loss, sizeof(float), 1, f) != 1 ||
        fread(&bs->learning_rate, sizeof(float), 1, f) != 1 ||
        fread(bs->mlp.w1, sizeof(float), BODY_HIDDEN_DIM * BODY_INPUT_DIM, f) != BODY_HIDDEN_DIM * BODY_INPUT_DIM ||
        fread(bs->mlp.b1, sizeof(float), BODY_HIDDEN_DIM, f) != BODY_HIDDEN_DIM ||
        fread(bs->mlp.w2, sizeof(float), BODY_HIDDEN_DIM, f) != BODY_HIDDEN_DIM ||
        fread(&bs->mlp.b2, sizeof(float), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    fclose(f);
    bs->initialized = 1;

    printf("[BodySense] Loaded: %d observations, loss=%.4f\n",
           bs->observations, bs->running_loss);

    return 1;
}

// ============================================================
// Debug
// ============================================================

void print_body_state(BodyState* state) {
    printf("BodyState:\n");
    printf("  entropy:    %.3f\n", state->entropy);
    printf("  novelty:    %.3f\n", state->novelty);
    printf("  arousal:    %.3f\n", state->arousal);
    printf("  valence:    %.3f\n", state->valence);
    printf("  unique:     %.3f\n", state->unique_ratio);
    printf("  quality:    %.3f\n", state->quality);
    printf("  expert:     %d (temp=%.2f)\n", state->expert_id, state->expert_temp);
}

void print_regulation(RegulationResult* result) {
    static const char* expert_names[] = {"structural", "semantic", "creative", "precise"};

    printf("Regulation:\n");
    printf("  boredom:    %.3f%s\n", result->boredom,
           result->boredom > 0.6f ? " [HIGH]" : "");
    printf("  overwhelm:  %.3f%s\n", result->overwhelm,
           result->overwhelm > 0.7f ? " [HIGH]" : "");
    printf("  stuck:      %.3f%s\n", result->stuck,
           result->stuck > 0.6f ? " [HIGH]" : "");
    printf("  quality:    %.3f (predicted)\n", result->predicted_quality);
    printf("  temp:       %.3f\n", result->temperature);
    printf("  expert:     %s\n", expert_names[result->expert_id % 4]);
}

void print_body_sense_stats(BodySense* bs) {
    float avg_b, avg_o, avg_s;
    body_get_stats(bs, &avg_b, &avg_o, &avg_s);

    printf("\n=== BodySense Stats ===\n");
    printf("Observations: %d\n", bs->observations);
    printf("Running loss: %.4f\n", bs->running_loss);
    printf("Last loss:    %.4f\n", bs->last_loss);
    printf("History:      %d entries\n", bs->history_count);
    printf("Avg boredom:  %.3f\n", avg_b);
    printf("Avg overwhelm: %.3f\n", avg_o);
    printf("Avg stuck:    %.3f\n", avg_s);
    printf("=======================\n\n");
}
