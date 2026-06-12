/* larynx.h — C interface to the Larynx (Tongue↔Soul bridge, inside vagus).
 *
 * The larynx measures the texture of a voice's token stream — entropy, pattern
 * strength (RRPRAM-lite trigram tracking) — and computes a blend factor α from
 * that texture plus the field's prophecy-debt and calendar-dissonance. In the
 * duo it carries Janus's spoken texture across to Resonance, so the inner voice
 * shapes its reply to HOW the outer voice just spoke, not only to the words.
 *
 * Implemented in vagus/vagus.zig; link with libvagus.
 */
#ifndef LARYNX_H
#define LARYNX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void  larynx_ingest_token(uint32_t token);   /* feed one emitted token */
void  larynx_reset(void);                     /* new utterance / conversation */
float larynx_get_entropy(void);               /* 0 = predictable, 1 = chaotic */
float larynx_get_pattern_strength(void);      /* recurring-pattern strength */
float larynx_get_alpha(void);                 /* current blend factor */
float larynx_compute_alpha(float prophecy_debt, float calendar_dissonance);
void  larynx_get_signal(float* out_entropy, float* out_pattern,
                        float* out_coherence, float* out_alpha);
int   larynx_get_recent_tokens(uint32_t* out, int max_tokens);

#define LARYNX_INGEST(tok) larynx_ingest_token((uint32_t)(tok))
#define LARYNX_ALPHA()     larynx_get_alpha()

#ifdef __cplusplus
}
#endif

#endif /* LARYNX_H */
