// arianna_dsl.c — DSL Integration for Arianna Generation
// ═══════════════════════════════════════════════════════════════════════════════

#include "arianna_dsl.h"
#include "identity_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

void dsl_init(void) {
    am_init();
    schumann_init();
}

int dsl_exec(const char* script) {
    return am_exec(script);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILD CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

DSL_GenerationConfig dsl_build_config(void) {
    DSL_GenerationConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    AM_State* s = am_get_state();

    // Temperature from velocity
    cfg.temperature = s->effective_temp;

    // Sampling params
    cfg.top_p = 0.95f;
    cfg.top_k = 50;
    cfg.repetition_penalty = 1.2f;

    // Prophecy — deeper prophecy strengthens destiny pull
    cfg.lookahead = s->prophecy;
    float prophecy_scale = 1.0f + (s->prophecy - 7) * 0.02f;  // default=7 → 1.0
    if (prophecy_scale < 0.5f) prophecy_scale = 0.5f;
    if (prophecy_scale > 2.0f) prophecy_scale = 2.0f;
    cfg.destiny_bias = s->destiny * prophecy_scale;

    // Suffering modulation
    cfg.pain_dampen = s->pain * 0.3f;
    cfg.tension_focus = s->tension * 0.2f;
    cfg.dissonance = s->dissonance;

    // Wormhole
    cfg.wormhole_chance = s->wormhole;
    cfg.wormhole_active = 0;

    // Attention physics
    cfg.attend_focus = s->attend_focus;
    cfg.attend_spread = s->attend_spread;

    // Tunneling (dissonance-gated)
    cfg.tunnel_threshold = s->tunnel_threshold;
    cfg.tunnel_chance = s->tunnel_chance;
    cfg.tunnel_skip_max = s->tunnel_skip_max;

    // Laws of nature
    cfg.entropy_floor = s->entropy_floor;
    cfg.resonance_ceiling = s->resonance_ceiling;
    cfg.emergence_threshold = s->emergence_threshold;

    // Calendar — modulated by birthday dissonance
    cfg.calendar_drift = s->calendar_drift;
    {
        time_t now = time(NULL);
        struct tm* tm_now = localtime(&now);
        if (tm_now) {
            float bd = identity_birthday_dissonance(
                tm_now->tm_year + 1900, tm_now->tm_mon + 1, tm_now->tm_mday);
            cfg.calendar_drift *= (1.0f + bd);  // dissonance amplifies drift
        }
    }

    // Cloud (will be set by dsl_apply_cloud)
    cfg.needs_care = 0;
    cfg.needs_warmth = 0;
    cfg.emotion_temp_bias = 0.0f;

    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════════
// APPLY TO LOGITS
// ═══════════════════════════════════════════════════════════════════════════════

void dsl_apply_to_logits(float* logits, int vocab_size,
                         const DSL_GenerationConfig* cfg) {
    // 1. Apply destiny bias
    if (cfg->destiny_bias > 0.01f) {
        dsl_apply_destiny(logits, vocab_size, cfg->destiny_bias);
    }

    // 2. Apply pain dampening (reduce extremes)
    if (cfg->pain_dampen > 0.01f) {
        float dampen = 1.0f - cfg->pain_dampen;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= dampen;
        }
    }

    // 3. Apply tension focus (sharpen distribution)
    if (cfg->tension_focus > 0.01f) {
        float sharpen = 1.0f + cfg->tension_focus;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= sharpen;
        }
    }

    // 4. Apply emotion temperature bias
    if (fabsf(cfg->emotion_temp_bias) > 0.01f) {
        float scale = 1.0f / (1.0f + cfg->emotion_temp_bias);
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= scale;
        }
    }

    // 5. Attention physics: focus sharpens, spread flattens
    // Net effect = (focus - spread) controls distribution peakedness
    float attend_net = cfg->attend_focus - cfg->attend_spread;
    if (fabsf(attend_net) > 0.01f) {
        // Find mean logit for centering
        float mean = 0.0f;
        for (int i = 0; i < vocab_size; i++) mean += logits[i];
        mean /= vocab_size;

        // Scale deviations from mean: >0 sharpens, <0 flattens
        float scale = 1.0f + attend_net * 0.5f;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = mean + (logits[i] - mean) * scale;
        }
    }

    // 6. Dissonance: inject noise proportional to symmetry-break
    // Higher dissonance = more chaotic, less predictable output
    if (cfg->dissonance > 0.05f) {
        for (int i = 0; i < vocab_size; i++) {
            float noise = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 2.0f;
            logits[i] += noise * cfg->dissonance * 0.5f;
        }
    }

    // 7. LAW: Entropy floor — prevent distribution from collapsing
    // If max logit dominates too much, flatten toward uniform
    if (cfg->entropy_floor > 0.01f) {
        float max_logit = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        // Compute approximate peakedness: ratio of max to sum-of-exp
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            float e = expf(logits[i] - max_logit);
            sum_exp += e;
        }
        float max_prob = 1.0f / sum_exp;  // probability of top token
        // If top token dominates beyond (1 - entropy_floor), flatten
        float dominance_limit = 1.0f - cfg->entropy_floor;
        if (max_prob > dominance_limit && dominance_limit > 0.0f) {
            // Reduce contrast: shrink logits toward their mean
            float flatten = dominance_limit / max_prob;
            float mean = 0.0f;
            for (int i = 0; i < vocab_size; i++) mean += logits[i];
            mean /= vocab_size;
            for (int i = 0; i < vocab_size; i++) {
                logits[i] = mean + (logits[i] - mean) * flatten;
            }
        }
    }

    // 8. LAW: Resonance ceiling — cap peak probability
    // Prevents any single token from having probability > ceiling
    if (cfg->resonance_ceiling < 0.99f && cfg->resonance_ceiling > 0.0f) {
        float max_logit = logits[0];
        int max_idx = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_idx = i;
            }
        }
        // Compute second highest for reference
        float second = -1e30f;
        for (int i = 0; i < vocab_size; i++) {
            if (i != max_idx && logits[i] > second) second = logits[i];
        }
        // If gap is too large, compress the top logit
        // Target: max_logit such that softmax(max) / (softmax(max) + (V-1)*softmax(second)) ≈ ceiling
        // Approximation: cap the gap between max and second
        float max_gap = -logf(1.0f / cfg->resonance_ceiling - 1.0f) + logf((float)(vocab_size - 1));
        float current_gap = max_logit - second;
        if (current_gap > max_gap && current_gap > 0.0f) {
            logits[max_idx] = second + max_gap;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOUD INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

void dsl_apply_cloud(DSL_GenerationConfig* cfg, const CloudResponse* cloud) {
    cfg->needs_care = cloud_needs_care(cloud);
    cfg->needs_warmth = cloud_needs_warmth(cloud);
    cfg->emotion_temp_bias = cloud_temperature_bias(cloud);

    // If needs care, reduce wormhole chance (be more stable)
    if (cfg->needs_care) {
        cfg->wormhole_chance *= 0.5f;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORMHOLE — creative skip
// ═══════════════════════════════════════════════════════════════════════════════

int dsl_check_wormhole(const DSL_GenerationConfig* cfg) {
    if (cfg->wormhole_chance <= 0.0f) return 0;

    float r = (float)rand() / (float)RAND_MAX;
    if (r < cfg->wormhole_chance) {
        // Wormhole activated! Skip 1-3 tokens
        int skip = 1 + (rand() % 3);
        return skip;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUNNELING — dissonance-gated skip
// ═══════════════════════════════════════════════════════════════════════════════

int dsl_check_tunneling(const DSL_GenerationConfig* cfg) {
    // Tunneling only fires when dissonance exceeds threshold
    if (cfg->dissonance < cfg->tunnel_threshold) return 0;
    if (cfg->tunnel_chance <= 0.0f) return 0;

    float r = (float)rand() / (float)RAND_MAX;
    if (r < cfg->tunnel_chance) {
        // Tunnel: skip 1 to tunnel_skip_max tokens
        int max_skip = cfg->tunnel_skip_max;
        if (max_skip < 1) max_skip = 1;
        int skip = 1 + (rand() % max_skip);
        return skip;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP PHYSICS
// ═══════════════════════════════════════════════════════════════════════════════

void dsl_step(float dt) {
    am_step(dt);
    schumann_step(dt);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPERATURE
// ═══════════════════════════════════════════════════════════════════════════════

float dsl_get_temperature(const DSL_GenerationConfig* cfg) {
    float temp = cfg->temperature;

    // Emotion bias
    temp += cfg->emotion_temp_bias;

    // Clamp
    if (temp < 0.1f) temp = 0.1f;
    if (temp > 2.0f) temp = 2.0f;

    return temp;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESTINY — bias toward most probable
// ═══════════════════════════════════════════════════════════════════════════════

void dsl_apply_destiny(float* logits, int vocab_size, float destiny) {
    // Find max logit
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // Boost tokens close to max, suppress others
    for (int i = 0; i < vocab_size; i++) {
        float diff = max_logit - logits[i];
        float suppress = diff * destiny * 0.5f;
        logits[i] -= suppress;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPHECY DEBT
// ═══════════════════════════════════════════════════════════════════════════════

float dsl_compute_prophecy_debt(const float* logits, int chosen_token, int vocab_size) {
    // Debt = how far from the most probable we chose
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float chosen_logit = logits[chosen_token];
    float diff = max_logit - chosen_logit;

    // Normalize to 0-1 range
    return diff > 0.0f ? diff / (diff + 1.0f) : 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CALENDAR DRIFT
// ═══════════════════════════════════════════════════════════════════════════════

float dsl_get_calendar_drift(void) {
    AM_State* s = am_get_state();
    return s->calendar_drift;
}

void dsl_apply_calendar_drift(float* logits, int vocab_size,
                              float drift, const int* time_tokens, int n_time_tokens) {
    // Boost/suppress time-related tokens based on drift
    // Positive drift = future bias, negative = past bias
    float bias = drift * 0.01f;

    for (int i = 0; i < n_time_tokens; i++) {
        int tok = time_tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            logits[tok] += bias;
        }
    }
}
