/*
 * larynx.h — The Tongue↔Soul Connection
 * ═══════════════════════════════════════════════════════════════════════════════
 * הגרון — מחבר בין הלשון לנשמה
 * The larynx: where thought becomes voice, where voice becomes identity.
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Part of vagus.zig — Zig implementation, C interface.
 *
 * Data flow:
 *   Tongue (500M, Qwen2.5 0.5B) → TEXT OUTWARD + larynx_ingest_token() → RRPRAM-lite → Soul (36M)
 *   Tongue is the MAIN BRAIN. Larynx carries its output inward to Soul for reflection.
 *
 * Key concepts:
 *   - Pattern recognition without training (trigram statistics)
 *   - Entropy measurement (predictability of output)
 *   - Alpha: dynamic blend of pattern vs semantic attention
 *     α = f(entropy, prophecy_debt, calendar_dissonance)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef LARYNX_H
#define LARYNX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CORE API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Ingest token from Tongue output */
void larynx_ingest_token(uint32_t token);

/* Reset Larynx state (call on new conversation) */
void larynx_reset(void);

/* ═══════════════════════════════════════════════════════════════════════════════
 * METRICS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Get current entropy (0 = predictable, 1 = chaotic) */
float larynx_get_entropy(void);

/* Get pattern strength (how strong are recurring patterns) */
float larynx_get_pattern_strength(void);

/* Get alpha blend factor (0.1 = semantic, 0.9 = pattern) */
float larynx_get_alpha(void);

/* Compute alpha based on external state
 *   prophecy_debt: from AM_State (higher = more pattern focus)
 *   calendar_dissonance: from identity (higher = more semantic focus)
 */
float larynx_compute_alpha(float prophecy_debt, float calendar_dissonance);

/* Get full signal (all metrics at once) */
void larynx_get_signal(float* out_entropy, float* out_pattern,
                       float* out_coherence, float* out_alpha);

/* ═══════════════════════════════════════════════════════════════════════════════
 * TOKEN HISTORY
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Get recent tokens for Soul processing
 * Returns number of tokens copied (up to max_tokens) */
int larynx_get_recent_tokens(uint32_t* out, int max_tokens);

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONVENIENCE MACROS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Use in generation loop after each token */
#define LARYNX_INGEST(tok) larynx_ingest_token((uint32_t)(tok))

/* Get alpha for hybrid attention blend */
#define LARYNX_ALPHA() larynx_get_alpha()

#ifdef __cplusplus
}
#endif

#endif /* LARYNX_H */
