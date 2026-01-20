// arianna_dsl.c — DSL Integration for Arianna Generation
// ═══════════════════════════════════════════════════════════════════════════════

#include "arianna_dsl.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

    // Prophecy
    cfg.lookahead = s->prophecy;
    cfg.destiny_bias = s->destiny;

    // Suffering modulation
    cfg.pain_dampen = s->pain * 0.3f;
    cfg.tension_focus = s->tension * 0.2f;

    // Wormhole
    cfg.wormhole_chance = s->wormhole;
    cfg.wormhole_active = 0;

    // Calendar
    cfg.calendar_drift = s->calendar_drift;

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
