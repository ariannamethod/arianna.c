/*
 * mood.c - Stanley-style mood routing implementation
 *
 * "Each mood is not a mask but a lens"
 */

#include "mood.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// ============================================================
// Initialization
// ============================================================

void init_mood_router(MoodRouter* router, float temperature) {
    memset(router, 0, sizeof(MoodRouter));
    router->temperature = temperature > 0.0f ? temperature : 1.0f;

    // Start with equal mix
    for (int i = 0; i < NUM_MOODS; i++) {
        router->mix[i] = 1.0f / NUM_MOODS;
    }
    router->dominant = MOOD_CALM;
}

// ============================================================
// Signal → Mood routing
// ============================================================

void route_signals_to_moods(MoodRouter* router, Signals* signals) {
    float max_score = -1e9f;

    // Compute raw scores for each mood
    for (int m = 0; m < NUM_MOODS; m++) {
        const MoodScoring* scoring = &MOOD_SCORING[m];

        float score = 0.0f;
        score += scoring->arousal_weight * signals->arousal;
        score += scoring->entropy_weight * signals->entropy;
        score += scoring->tension_weight * signals->tension;
        score += scoring->warmth_weight * signals->warmth;
        score += scoring->focus_weight * signals->focus;
        score += scoring->recursion_weight * signals->recursion_depth;
        score += scoring->resonance_weight * signals->resonance;
        score += scoring->novelty_weight * signals->novelty;

        router->scores[m] = score;

        if (score > max_score) {
            max_score = score;
            router->dominant = (Mood)m;
        }
    }

    // Softmax with temperature
    float sum = 0.0f;
    for (int m = 0; m < NUM_MOODS; m++) {
        router->mix[m] = expf((router->scores[m] - max_score) / router->temperature);
        sum += router->mix[m];
    }

    // Normalize
    for (int m = 0; m < NUM_MOODS; m++) {
        router->mix[m] /= sum;
    }

    // Compute effective temperature bias from mix
    router->effective_temp_bias = 0.0f;
    for (int m = 0; m < NUM_MOODS; m++) {
        router->effective_temp_bias += router->mix[m] * MOOD_PROFILES[m].temperature_bias;
    }
}

// ============================================================
// Accessors
// ============================================================

float get_temperature_bias(MoodRouter* router) {
    return router->effective_temp_bias;
}

const MoodProfile* get_dominant_profile(MoodRouter* router) {
    return &MOOD_PROFILES[router->dominant];
}

// ============================================================
// Debug output
// ============================================================

void print_mood_state(MoodRouter* router) {
    printf("Mood state:\n");
    printf("  Dominant: %s (%.1f%%)\n",
           MOOD_NAMES[router->dominant],
           router->mix[router->dominant] * 100.0f);

    printf("  Mix: ");
    for (int m = 0; m < NUM_MOODS; m++) {
        if (router->mix[m] > 0.05f) {  // Only show significant moods
            printf("%s:%.0f%% ", MOOD_NAMES[m], router->mix[m] * 100.0f);
        }
    }
    printf("\n");

    printf("  Temp bias: %+.2f\n", router->effective_temp_bias);
}

// ============================================================
// Mood → Shard mapping
// ============================================================

/*
 * If shards are named after moods (calm.bin, intense.bin, etc.),
 * map mood mix directly to shard mix.
 *
 * If shards have other names, this finds best match by name.
 */
void mood_to_shard_mix(MoodRouter* router, DeltaBank* bank) {
    if (bank->n_shards == 0) return;

    // Try to match shard names to moods
    for (int s = 0; s < bank->n_shards; s++) {
        bank->mix[s] = 0.0f;

        // Check if shard name contains a mood name
        for (int m = 0; m < NUM_MOODS; m++) {
            if (strstr(bank->shards[s].name, MOOD_NAMES[m]) != NULL) {
                bank->mix[s] = router->mix[m];
                break;
            }
        }
    }

    // Normalize if any matches found
    float total = 0.0f;
    for (int s = 0; s < bank->n_shards; s++) {
        total += bank->mix[s];
    }

    if (total > 0.0f) {
        for (int s = 0; s < bank->n_shards; s++) {
            bank->mix[s] /= total;
        }
    } else {
        // No mood matches - use equal weights
        for (int s = 0; s < bank->n_shards; s++) {
            bank->mix[s] = 1.0f / bank->n_shards;
        }
    }

    bank->cache_valid = 0;
}

// ============================================================
// Advanced: Mood transitions (Stanley's drift momentum)
// ============================================================

/*
 * Smooth mood transitions with momentum.
 * Prevents jarring shifts between moods.
 */
void update_mood_with_momentum(MoodRouter* router, Signals* signals, float momentum) {
    // Store old mix
    float old_mix[NUM_MOODS];
    memcpy(old_mix, router->mix, sizeof(old_mix));

    // Compute new mix from signals
    route_signals_to_moods(router, signals);

    // Blend with momentum: new = momentum * old + (1-momentum) * new
    for (int m = 0; m < NUM_MOODS; m++) {
        router->mix[m] = momentum * old_mix[m] + (1.0f - momentum) * router->mix[m];
    }

    // Renormalize
    float sum = 0.0f;
    for (int m = 0; m < NUM_MOODS; m++) {
        sum += router->mix[m];
    }
    for (int m = 0; m < NUM_MOODS; m++) {
        router->mix[m] /= sum;
    }

    // Update dominant
    float max_mix = 0.0f;
    for (int m = 0; m < NUM_MOODS; m++) {
        if (router->mix[m] > max_mix) {
            max_mix = router->mix[m];
            router->dominant = (Mood)m;
        }
    }
}

// ============================================================
// Mood-based temperature adjustment for sampling
// ============================================================

float adjust_temperature_by_mood(MoodRouter* router, float base_temp) {
    // Clamp adjusted temperature to reasonable range
    float adjusted = base_temp + router->effective_temp_bias;
    if (adjusted < 0.1f) adjusted = 0.1f;
    if (adjusted > 2.0f) adjusted = 2.0f;
    return adjusted;
}
