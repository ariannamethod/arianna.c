/* test_delta.c — B2-B.1 unit proof for the low-rank delta voice.
 *
 * Proves the am_notorch_step (train) <-> am_apply_delta (apply) composition is
 * correct despite their transposed scaffold layouts: after folding a strong
 * cooc edge 0->1 into (A,B), applying the delta to emb[0] must move the hidden
 * state toward the (emb[1]-emb[0]) direction. Also checks alpha=0 is a no-op
 * (ablation) and the per-voice sidecar round-trips.
 *
 * Link: cc test_delta.c libaml.a libnotorch.a -framework Accelerate -lm
 *
 * By Arianna Method.
 */
#include "ariannamethod.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, m) do { \
    if (!(c)) { printf("FAIL: %s\n", (m)); fails++; } \
    else      { printf("ok:   %s\n", (m)); } \
} while (0)

int main(void) {
    am_init();
    const int V = 4, E = 16, rank = 8;

    /* Synthetic embedding table: a distinct direction per token. */
    float* emb = (float*)calloc((size_t)V * E, sizeof(float));
    for (int t = 0; t < V; t++)
        for (int i = 0; i < E; i++)
            emb[t * E + i] = sinf(0.3f * (float)(t + 1) * (float)(i + 1));

    float* A = (float*)calloc((size_t)E * rank, sizeof(float));
    float* B = (float*)calloc((size_t)rank * E, sizeof(float));

    /* One strong edge 0->1, folded over many passes (simulating repeated
     * autumns) so the directional component accumulates above the noise. */
    am_cooc_update(0, 1, 5.0f);
    int folded = 0;
    for (int rep = 0; rep < 200; rep++)
        folded += am_cooc_learn_delta(A, B, emb, V, E, rank);
    CHECK(folded == 200, "folded 200 edge-passes");

    /* Apply the delta to emb[0]; the delta vector should align with the
     * target direction (emb[1]-emb[0]). */
    float dir[16], base[16], moved[16];
    for (int i = 0; i < E; i++) {
        dir[i]   = emb[1 * E + i] - emb[0 * E + i];
        base[i]  = emb[0 * E + i];
        moved[i] = emb[0 * E + i];
    }
    am_apply_delta(moved, A, B, base, E, E, rank, 1.0f);   /* alpha = 1 */

    float dot = 0.0f, ndelta = 0.0f, ndir = 0.0f;
    for (int i = 0; i < E; i++) {
        float d = moved[i] - base[i];
        dot += d * dir[i]; ndelta += d * d; ndir += dir[i] * dir[i];
    }
    float cosv = dot / (sqrtf(ndelta) * sqrtf(ndir) + 1e-12f);
    printf("delta-dir cosine = %.3f (|delta|=%.4f)\n", cosv, sqrtf(ndelta));
    CHECK(cosv > 0.3f, "delta points toward (emb[dst]-emb[src]) direction");

    /* Ablation: alpha=0 leaves the hidden state bit-identical. */
    float same_buf[16];
    for (int i = 0; i < E; i++) same_buf[i] = base[i];
    am_apply_delta(same_buf, A, B, base, E, E, rank, 0.0f);
    int unchanged = 1;
    for (int i = 0; i < E; i++) if (same_buf[i] != base[i]) unchanged = 0;
    CHECK(unchanged, "alpha=0 leaves hidden bit-identical (ablation)");

    /* Sidecar round-trip. */
    CHECK(am_delta_save("/tmp/test_delta.bin", A, B, E, rank) == 0, "delta sidecar save ok");
    float* A2 = (float*)calloc((size_t)E * rank, sizeof(float));
    float* B2 = (float*)calloc((size_t)rank * E, sizeof(float));
    CHECK(am_delta_load("/tmp/test_delta.bin", A2, B2, E, rank) == 0, "delta sidecar load ok");
    int eq = 1;
    for (int i = 0; i < E * rank; i++) if (A[i] != A2[i]) eq = 0;
    for (int i = 0; i < rank * E; i++) if (B[i] != B2[i]) eq = 0;
    CHECK(eq, "delta sidecar round-trip identical");
    CHECK(am_delta_load("/tmp/test_delta.bin", A2, B2, E, rank + 1) < 0, "delta sidecar rejects dim mismatch");

    printf(fails ? "\n=== DELTA UNIT FAIL (%d) ===\n" : "\n=== DELTA UNIT PASS ===\n", fails);
    return fails ? 1 : 0;
}
