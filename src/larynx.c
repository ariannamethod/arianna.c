/*
 * larynx.c — Pure C fallback for Larynx (Tongue↔Soul bridge).
 *
 * Mirrors the Zig implementation in vagus.zig when Zig bridge is not linked.
 * Implements real alpha computation, token history, and n-gram tracking
 * instead of returning hardcoded defaults.
 */

#include "larynx.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE — mirrors Zig Larynx struct
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TRIGRAM_BUF 64

static struct {
    uint32_t history[TRIGRAM_BUF];
    int      pos;
    int      count;
    uint32_t bigram_hits;
    uint32_t trigram_hits;
    uint32_t total_tokens;
    float    entropy;
    float    pattern_strength;
    float    trigram_coherence;
    float    alpha;
} L = {
    .pos = 0, .count = 0,
    .bigram_hits = 0, .trigram_hits = 0, .total_tokens = 0,
    .entropy = 0.5f, .pattern_strength = 0.5f,
    .trigram_coherence = 0.5f, .alpha = 0.5f,
};

/* ═══════════════════════════════════════════════════════════════════════════════
 * N-GRAM TRACKING
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void update_bigram_stats(void) {
    if (L.count < 4) return;
    int curr_pos = (L.pos + TRIGRAM_BUF - 1) % TRIGRAM_BUF;
    int prev_pos = (L.pos + TRIGRAM_BUF - 2) % TRIGRAM_BUF;
    uint32_t curr = L.history[curr_pos];
    uint32_t prev = L.history[prev_pos];

    for (int i = 0; i + 1 < L.count - 2; i++) {
        int idx = (L.pos + TRIGRAM_BUF - L.count + i) % TRIGRAM_BUF;
        int idx_next = (idx + 1) % TRIGRAM_BUF;
        if (L.history[idx] == prev && L.history[idx_next] == curr) {
            L.bigram_hits++;
            break;
        }
    }
}

static void update_trigram_stats(void) {
    if (L.count < 6) return;
    int p0 = (L.pos + TRIGRAM_BUF - 3) % TRIGRAM_BUF;
    int p1 = (L.pos + TRIGRAM_BUF - 2) % TRIGRAM_BUF;
    int p2 = (L.pos + TRIGRAM_BUF - 1) % TRIGRAM_BUF;
    uint32_t t0 = L.history[p0], t1 = L.history[p1], t2 = L.history[p2];

    for (int i = 0; i + 2 < L.count - 3; i++) {
        int idx = (L.pos + TRIGRAM_BUF - L.count + i) % TRIGRAM_BUF;
        if (L.history[idx] == t0 &&
            L.history[(idx + 1) % TRIGRAM_BUF] == t1 &&
            L.history[(idx + 2) % TRIGRAM_BUF] == t2) {
            L.trigram_hits++;
            break;
        }
    }
}

static void compute_metrics(void) {
    if (L.total_tokens < 4) return;
    float total = (float)L.total_tokens;
    float bigram_rate  = (float)L.bigram_hits  / total;
    float trigram_rate = (float)L.trigram_hits / total;

    /* High hit rate = low entropy (predictable) — matches Zig */
    float raw = bigram_rate * 2.0f + trigram_rate * 3.0f;
    if (raw < 0.0f) raw = 0.0f;
    if (raw > 1.0f) raw = 1.0f;
    L.entropy = 1.0f - raw;

    /* Pattern strength */
    L.pattern_strength = trigram_rate * 5.0f;
    if (L.pattern_strength > 1.0f) L.pattern_strength = 1.0f;
    if (L.pattern_strength < 0.0f) L.pattern_strength = 0.0f;

    /* Trigram coherence */
    if (L.bigram_hits > 0) {
        L.trigram_coherence = (float)L.trigram_hits / (float)L.bigram_hits;
        if (L.trigram_coherence > 1.0f) L.trigram_coherence = 1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PUBLIC API — matches larynx.h contract
 * ═══════════════════════════════════════════════════════════════════════════════ */

void larynx_ingest_token(uint32_t token) {
    L.history[L.pos] = token;
    L.pos = (L.pos + 1) % TRIGRAM_BUF;
    if (L.count < TRIGRAM_BUF) L.count++;
    L.total_tokens++;

    if (L.count >= 2) update_bigram_stats();
    if (L.count >= 3) update_trigram_stats();
    compute_metrics();
}

void larynx_reset(void) {
    memset(L.history, 0, sizeof(L.history));
    L.pos = 0;
    L.count = 0;
    L.bigram_hits = 0;
    L.trigram_hits = 0;
    L.total_tokens = 0;
    L.entropy = 0.5f;
    L.pattern_strength = 0.5f;
    L.trigram_coherence = 0.5f;
    L.alpha = 0.5f;
}

float larynx_get_entropy(void) {
    return L.entropy;
}

float larynx_get_pattern_strength(void) {
    return L.pattern_strength;
}

float larynx_get_alpha(void) {
    return L.alpha;
}

float larynx_compute_alpha(float prophecy_debt, float calendar_dissonance) {
    /* Mirrors vagus.zig Larynx.computeAlpha exactly:
     *   base  = 0.5
     *   base += entropy * 0.2
     *   base += prophecy_debt * 0.15
     *   base -= calendar_dissonance * 0.1
     *   clamp [0.1, 0.9]
     */
    float base = 0.5f;
    base += L.entropy * 0.2f;
    base += prophecy_debt * 0.15f;
    base -= calendar_dissonance * 0.1f;
    if (base < 0.1f) base = 0.1f;
    if (base > 0.9f) base = 0.9f;
    L.alpha = base;
    return L.alpha;
}

void larynx_get_signal(float* out_entropy, float* out_pattern,
                       float* out_coherence, float* out_alpha) {
    if (out_entropy)   *out_entropy   = L.entropy;
    if (out_pattern)   *out_pattern   = L.pattern_strength;
    if (out_coherence) *out_coherence = L.trigram_coherence;
    if (out_alpha)     *out_alpha     = L.alpha;
}

int larynx_get_recent_tokens(uint32_t* out, int max_tokens) {
    if (!out || max_tokens <= 0) return 0;
    int n = L.count < max_tokens ? L.count : max_tokens;
    for (int i = 0; i < n; i++) {
        int idx = (L.pos + TRIGRAM_BUF - n + i) % TRIGRAM_BUF;
        out[i] = L.history[idx];
    }
    return n;
}
