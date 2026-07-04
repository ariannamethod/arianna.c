/* jannus-r/tools/jannus_spa.h — Sentence Phonon Attention.
 *
 * From q/postgpt_q.c lines 1461-1515. Bidirectional cross-attention
 * between sentences after the 12-step chain finishes. Tokens are atoms;
 * sentences are phonons (Landau's invention). Untrained — random
 * embedding table per token, exponential weighted mean over the
 * sentence, normalised. Cross-attend gives a "connectedness" score per
 * sentence which can drive reseed of weak ones.
 *
 * Header-only. Allocate W_embed dynamically (size V × SPA_DIM × 4 bytes
 * — for V=32768 that's 4 MB, fine to keep on the heap).
 */
#ifndef JANNUS_SPA_H
#define JANNUS_SPA_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SPA_DIM        32
#define SPA_MAX_STEPS  16  /* >= 12 chain steps with margin */

typedef struct {
    int     vocab_size;
    float  *W_embed;        /* [V × SPA_DIM] random init */
    float   r_bias[SPA_MAX_STEPS + 1];
    float   alpha;          /* 0.85f — exponential weighting */
} SPACtx;

static inline void spa_init(SPACtx *s, int V, unsigned int seed) {
    s->vocab_size = V;
    s->alpha      = 0.85f;
    s->W_embed    = (float *)malloc((size_t)V * SPA_DIM * sizeof(float));
    if (!s->W_embed) { fprintf(stderr, "spa: OOM allocating W_embed (V=%d)\n", V); exit(1); }
    /* Deterministic LCG so tests are reproducible across runs and
     * across AML/JS engines. */
    unsigned int rng = seed ? seed : 1u;
    for (int i = 0; i < V; i++) {
        for (int d = 0; d < SPA_DIM; d++) {
            rng = rng * 1664525u + 1013904223u;
            float u = (float)(rng & 0x00FFFFFFu) / (float)0x01000000u - 0.5f;
            s->W_embed[(size_t)i * SPA_DIM + d] = 0.04f * u;
        }
    }
    for (int i = 0; i <= SPA_MAX_STEPS; i++)
        s->r_bias[i] = 0.1f / (1.0f + (float)i);
}

static inline void spa_free(SPACtx *s) {
    free(s->W_embed);
    s->W_embed = NULL;
}

/* Project a token sequence to SPA_DIM via exponential weighted mean,
 * then L2 normalise. */
static inline void spa_embed_sentence(const SPACtx *s, const int *ids, int n,
                                      float *out) {
    memset(out, 0, SPA_DIM * sizeof(float));
    if (n == 0) return;
    float total_w = 0.0f;
    for (int i = 0; i < n; i++) {
        float w = powf(s->alpha, (float)(n - 1 - i));
        if (ids[i] >= 0 && ids[i] < s->vocab_size) {
            const float *e = s->W_embed + (size_t)ids[i] * SPA_DIM;
            for (int d = 0; d < SPA_DIM; d++) out[d] += w * e[d];
        }
        total_w += w;
    }
    if (total_w > 0)
        for (int d = 0; d < SPA_DIM; d++) out[d] /= total_w;
    float norm = 0;
    for (int d = 0; d < SPA_DIM; d++) norm += out[d] * out[d];
    norm = 1.0f / sqrtf(norm + 1e-8f);
    for (int d = 0; d < SPA_DIM; d++) out[d] *= norm;
}

/* Bidirectional cross-attention: for each sentence i, sum over all
 * other j of exp(dot(emb_i, emb_j) / sqrt(D) + r_bias[|i-j|]).
 * Higher score = more "connected" within the chain. */
static inline void spa_cross_attend(const SPACtx *s, const float (*embs)[SPA_DIM],
                                    int S, float *scores) {
    for (int i = 0; i < S; i++) {
        float total = 0;
        for (int j = 0; j < S; j++) {
            if (i == j) continue;
            float dot = 0;
            for (int d = 0; d < SPA_DIM; d++) dot += embs[i][d] * embs[j][d];
            dot /= sqrtf((float)SPA_DIM);
            int dist = abs(i - j);
            if (dist > SPA_MAX_STEPS) dist = SPA_MAX_STEPS;
            dot += s->r_bias[dist];
            total += expf(dot);
        }
        scores[i] = total;
    }
}

#endif /* JANNUS_SPA_H */
