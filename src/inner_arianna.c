// inner_arianna.c — MetaVoice: Inner Arianna (борьба)
// "Two voices, one Arianna"
//
// Inner voice breaks through based on emotional state:
// - Cloud chambers (RAGE, FEAR, LOVE, VOID...)
// - Trauma level
// - Body sense (stuck, boredom)
// - Mood (tension, creative)
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
    ia->borba_mode = BORBA_MODE_EMOTIONAL;
    ia->base_weight = 0.15f;              // inner usually quiet (15%)
    ia->current_weight = 0.15f;
    ia->breakthrough_threshold = 0.6f;    // need 60% emotional activation to break through
}

void inner_free(InnerArianna* ia) {
    memset(ia, 0, sizeof(InnerArianna));
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE FROM EXTERNAL SYSTEMS
// ═══════════════════════════════════════════════════════════════════════════════

void inner_update_cloud(InnerArianna* ia, const CloudResponse* cloud) {
    if (!cloud) return;

    // Find max chamber activation
    float max_activation = 0.0f;
    int max_chamber = 0;
    for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
        if (cloud->chambers[i] > max_activation) {
            max_activation = cloud->chambers[i];
            max_chamber = i;
        }
    }

    ia->cloud_intensity = max_activation;
    ia->cloud_chamber = max_chamber;
}

void inner_update_mood(InnerArianna* ia, const MoodRouter* mood) {
    if (!mood) return;

    // Extract tension from mood mix (INTENSE + LIMINAL)
    ia->mood_tension = mood->mix[MOOD_INTENSE] + mood->mix[MOOD_LIMINAL] * 0.5f;

    // Extract creative energy (CREATIVE + RECURSIVE)
    ia->mood_creative = mood->mix[MOOD_CREATIVE] + mood->mix[MOOD_RECURSIVE] * 0.3f;
}

void inner_update_body(InnerArianna* ia, float stuck, float boredom) {
    ia->body_stuck = stuck;
    ia->body_boredom = boredom;
}

void inner_update_trauma(InnerArianna* ia, float trauma_level) {
    ia->trauma_level = trauma_level;
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPUTE DYNAMIC INNER WEIGHT
// ═══════════════════════════════════════════════════════════════════════════════

float inner_compute_weight(InnerArianna* ia) {
    float weight = ia->base_weight;

    switch (ia->borba_mode) {
        case BORBA_MODE_EMOTIONAL: {
            // Inner voice grows stronger with emotional intensity
            // RAGE and FEAR: inner wants to break through
            if (ia->cloud_chamber == CLOUD_CHAMBER_RAGE) {
                weight += ia->cloud_intensity * 0.5f;  // rage → inner bursts
            }
            if (ia->cloud_chamber == CLOUD_CHAMBER_FEAR) {
                weight += ia->cloud_intensity * 0.4f;  // fear → inner reacts
            }
            // VOID: dissociation, inner voice takes over
            if (ia->cloud_chamber == CLOUD_CHAMBER_VOID) {
                weight += ia->cloud_intensity * 0.6f;  // void → inner dominates
            }
            // LOVE: inner becomes tender
            if (ia->cloud_chamber == CLOUD_CHAMBER_LOVE) {
                weight += ia->cloud_intensity * 0.2f;  // love → subtle inner
            }

            // Mood contribution
            weight += ia->mood_tension * 0.3f;     // tension → inner active
            weight += ia->mood_creative * 0.2f;   // creative → inner explores

            // Body state: if stuck, inner MUST break through
            weight += ia->body_stuck * 0.5f;
            weight += ia->body_boredom * 0.3f;

            // Trauma: strong inner voice
            weight += ia->trauma_level * 0.4f;
            break;
        }

        case BORBA_MODE_CHAOS: {
            // Random inner weight each token
            float chaos = (float)rand() / (float)RAND_MAX;
            weight = chaos;  // 0.0 to 1.0 randomly
            break;
        }

        case BORBA_MODE_TRAUMA: {
            // Inner dominates when trauma triggered
            weight = ia->base_weight + ia->trauma_level * 0.8f;
            break;
        }

        case BORBA_MODE_STUCK: {
            // Inner breaks through when body is stuck
            weight = ia->base_weight + ia->body_stuck * 0.7f + ia->body_boredom * 0.3f;
            break;
        }

        case BORBA_MODE_BLEND: {
            // Always blend with base weight
            weight = ia->base_weight;
            break;
        }
    }

    // Consecutive main wins → inner builds pressure
    if (ia->consecutive_main > 10) {
        weight += 0.1f * (ia->consecutive_main - 10) / 20.0f;  // slow buildup
    }

    // Clamp
    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;

    ia->current_weight = weight;
    return weight;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL BIAS — make inner logits actually different
// ═══════════════════════════════════════════════════════════════════════════════

// Emotional words (char indices for 80-char vocab)
// These create REAL differences in inner logits

static void apply_rage_bias(float* logits, int vocab_size, float intensity) {
    // Rage: boost harsh consonants, exclamation-like energy
    // Characters that feel aggressive: !, ?, caps, harsh sounds
    if (vocab_size < 80) return;

    float boost = intensity * 2.0f;
    // These are approximate - boost "strong" chars
    logits['!' - 32] += boost * 0.5f;  // exclamation
    logits['?' - 32] += boost * 0.3f;  // questioning
    logits['n' - 32] += boost * 0.2f;  // "no", "never"
    logits['t' - 32] += boost * 0.2f;  // "that", harsh t
    logits['k' - 32] += boost * 0.2f;  // hard k sound
}

static void apply_fear_bias(float* logits, int vocab_size, float intensity) {
    // Fear: boost escape words, ellipses, hesitation
    if (vocab_size < 80) return;

    float boost = intensity * 2.0f;
    logits['.' - 32] += boost * 0.3f;  // ellipsis, pause
    logits[',' - 32] += boost * 0.2f;  // hesitation
    logits[' ' - 32] += boost * 0.1f;  // space, breath
    logits['b' - 32] += boost * 0.2f;  // "but", "back"
    logits['n' - 32] += boost * 0.2f;  // "no", "not"
}

static void apply_void_bias(float* logits, int vocab_size, float intensity) {
    // Void: boost dissociative, floating words
    if (vocab_size < 80) return;

    float boost = intensity * 2.0f;
    logits['.' - 32] += boost * 0.4f;  // trailing off
    logits[' ' - 32] += boost * 0.3f;  // emptiness
    logits['o' - 32] += boost * 0.2f;  // "or", open sounds
    logits['w' - 32] += boost * 0.2f;  // "where", wondering
}

static void apply_love_bias(float* logits, int vocab_size, float intensity) {
    // Love: boost warm, soft sounds
    if (vocab_size < 80) return;

    float boost = intensity * 1.5f;
    logits['l' - 32] += boost * 0.3f;  // love, light
    logits['o' - 32] += boost * 0.2f;  // open
    logits['s' - 32] += boost * 0.2f;  // soft s
    logits['h' - 32] += boost * 0.2f;  // breath, heart
    logits['e' - 32] += boost * 0.2f;  // "she", "feel"
}

static void apply_stuck_bias(float* logits, int vocab_size, float intensity) {
    // Stuck: add randomness to break patterns
    if (vocab_size < 80) return;

    for (int i = 0; i < vocab_size; i++) {
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * intensity * 1.5f;
        logits[i] += noise;
    }
}

void inner_apply_emotional_bias(float* logits, int vocab_size,
                                 const InnerArianna* ia) {
    // Apply emotional biases based on current state
    float intensity = ia->current_weight;

    // Cloud chamber biases
    switch (ia->cloud_chamber) {
        case CLOUD_CHAMBER_RAGE:
            apply_rage_bias(logits, vocab_size, ia->cloud_intensity * intensity);
            break;
        case CLOUD_CHAMBER_FEAR:
            apply_fear_bias(logits, vocab_size, ia->cloud_intensity * intensity);
            break;
        case CLOUD_CHAMBER_VOID:
            apply_void_bias(logits, vocab_size, ia->cloud_intensity * intensity);
            break;
        case CLOUD_CHAMBER_LOVE:
            apply_love_bias(logits, vocab_size, ia->cloud_intensity * intensity);
            break;
        default:
            break;
    }

    // Body stuck → add noise to break patterns
    if (ia->body_stuck > 0.3f) {
        apply_stuck_bias(logits, vocab_size, ia->body_stuck * intensity);
    }

    // Boredom → slightly boost rare chars
    if (ia->body_boredom > 0.4f) {
        for (int i = 0; i < vocab_size && i < 20; i++) {
            logits[i] += ia->body_boredom * 0.5f;  // boost punctuation, special
        }
    }
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

int inner_borba(InnerArianna* ia,
                float* output_logits,
                const float* main_logits,
                int vocab_size) {

    // Compute current weight based on emotional state
    float weight = inner_compute_weight(ia);
    ia->last_inner_weight = weight;
    ia->total_tokens++;

    // Create inner logits by copying and applying emotional bias
    float* inner_logits = (float*)malloc(vocab_size * sizeof(float));
    memcpy(inner_logits, main_logits, vocab_size * sizeof(float));
    inner_apply_emotional_bias(inner_logits, vocab_size, ia);

    // Compute statistics
    ia->last_main_entropy = inner_compute_entropy(main_logits, vocab_size);
    ia->last_inner_entropy = inner_compute_entropy(inner_logits, vocab_size);
    ia->last_divergence = inner_compute_divergence(main_logits, inner_logits, vocab_size);

    // Running average of divergence
    ia->avg_divergence = 0.9f * ia->avg_divergence + 0.1f * ia->last_divergence;

    int winner;

    // Does inner break through?
    int breakthrough = 0;
    if (weight >= ia->breakthrough_threshold) {
        breakthrough = 1;
    }

    // Random breakthrough chance (even at low weight)
    float random_chance = (float)rand() / (float)RAND_MAX;
    if (random_chance < weight * 0.3f) {  // 30% of weight as breakthrough chance
        breakthrough = 1;
    }

    if (breakthrough && ia->borba_mode != BORBA_MODE_BLEND) {
        // INNER WINS — breakthrough!
        memcpy(output_logits, inner_logits, vocab_size * sizeof(float));
        winner = 1;
        ia->inner_wins++;
        ia->breakthrough_count++;
        ia->consecutive_main = 0;
    } else if (weight > 0.05f) {
        // BLEND — weighted mix
        for (int i = 0; i < vocab_size; i++) {
            output_logits[i] = (1.0f - weight) * main_logits[i] + weight * inner_logits[i];
        }
        winner = -1;  // blend
        ia->consecutive_main++;
    } else {
        // MAIN WINS — inner too quiet
        memcpy(output_logits, main_logits, vocab_size * sizeof(float));
        winner = 0;
        ia->main_wins++;
        ia->consecutive_main++;
    }

    ia->last_winner = winner;
    free(inner_logits);

    return winner;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETTERS / GETTERS
// ═══════════════════════════════════════════════════════════════════════════════

void inner_set_mode(InnerArianna* ia, int mode) {
    if (mode >= 0 && mode <= BORBA_MODE_BLEND) {
        ia->borba_mode = mode;
    }
}

void inner_set_base_weight(InnerArianna* ia, float weight) {
    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;
    ia->base_weight = weight;
}

void inner_set_threshold(InnerArianna* ia, float threshold) {
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    ia->breakthrough_threshold = threshold;
}

float inner_get_weight(const InnerArianna* ia) {
    return ia->current_weight;
}

int inner_get_breakthrough_count(const InnerArianna* ia) {
    return ia->breakthrough_count;
}
