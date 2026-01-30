/*
 * larynx.c — Stub implementation until Zig bridge is ready.
 * Provides no-op defaults so arianna_dynamic links.
 */

#include "larynx.h"

static float g_entropy = 0.5f;
static float g_alpha = 0.5f;

void larynx_ingest_token(uint32_t token) {
    (void)token;
}

void larynx_reset(void) {
    g_entropy = 0.5f;
    g_alpha = 0.5f;
}

float larynx_get_entropy(void) {
    return g_entropy;
}

float larynx_get_pattern_strength(void) {
    return 0.0f;
}

float larynx_get_alpha(void) {
    return g_alpha;
}

float larynx_compute_alpha(float prophecy_debt, float calendar_dissonance) {
    (void)prophecy_debt;
    (void)calendar_dissonance;
    return g_alpha;
}

void larynx_get_signal(float* out_entropy, float* out_pattern,
                       float* out_coherence, float* out_alpha) {
    if (out_entropy) *out_entropy = g_entropy;
    if (out_pattern) *out_pattern = 0.0f;
    if (out_coherence) *out_coherence = 1.0f;
    if (out_alpha) *out_alpha = g_alpha;
}

int larynx_get_recent_tokens(uint32_t* out, int max_tokens) {
    (void)out;
    (void)max_tokens;
    return 0;
}
