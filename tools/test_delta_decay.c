/* test_delta_decay.c — B2-B.5 unit proof for the δ forgetting valve.
 *
 * am_cooc_learn_delta is a *converging* training step (am_notorch_step toward the
 * cooc-implied direction, clamped ±10), so δ self-bounds — it does NOT grow
 * without limit. am_delta_decay's real job is ADAPTIVITY: applied before each
 * autumn harvest it lets δ forget stale themes and track the recent dialogue.
 *
 * Proof: learn edge 0->1 for K autumns, then switch the cooc to edge 0->2 for K
 * autumns. With decay, δ rotates toward dir(emb2-emb0) (forgets 0->1); without
 * decay it lingers on the old direction. We assert decay aligns better with the
 * NEW direction after the switch.
 *
 * Link: cc -Iariannamethod/core test_delta_decay.c libaml.a libnotorch.a \
 *          -framework Accelerate -lm
 * By Arianna Method.
 */
#include "ariannamethod.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const int V = 4, E = 16, rank = 8, K = 14;

static float cos_to_dir(const float* A, const float* B, const float* emb, const float* dir) {
    float base[16], moved[16];
    for (int i = 0; i < E; i++) { base[i] = emb[i]; moved[i] = emb[i]; }
    am_apply_delta(moved, A, B, base, E, E, rank, 1.0f);
    float dot = 0, nd = 0, ng = 0;
    for (int i = 0; i < E; i++) { float d = moved[i] - base[i]; dot += d * dir[i]; nd += d * d; ng += dir[i] * dir[i]; }
    return dot / (sqrtf(nd) * sqrtf(ng) + 1e-12f);
}

/* Learn the currently-loaded cooc for K autumns into (A,B) with the given decay. */
static void learn_phase(float* A, float* B, const float* emb, float decay) {
    for (int a = 0; a < K; a++) { am_delta_decay(A, B, E, rank, decay); am_cooc_learn_delta(A, B, emb, V, E, rank); }
}

int main(void) {
    am_init();
    float* emb = (float*)calloc((size_t)V * E, sizeof(float));
    for (int t = 0; t < V; t++)
        for (int i = 0; i < E; i++)
            emb[t * E + i] = sinf(0.3f * (float)(t + 1) * (float)(i + 1));
    float dir02[16];
    for (int i = 0; i < E; i++) dir02[i] = emb[2 * E + i] - emb[0 * E + i];

    float *Ad = calloc((size_t)E*rank,4), *Bd = calloc((size_t)rank*E,4);  /* decay 0.9 */
    float *An = calloc((size_t)E*rank,4), *Bn = calloc((size_t)rank*E,4);  /* decay 1.0 */

    /* Phase 1: theme 0->1 */
    am_cooc_clear(); am_cooc_update(0, 1, 5.0f);
    learn_phase(Ad, Bd, emb, 0.9f);
    learn_phase(An, Bn, emb, 1.0f);

    /* Phase 2: theme switches to 0->2 */
    am_cooc_clear(); am_cooc_update(0, 2, 5.0f);
    learn_phase(Ad, Bd, emb, 0.9f);
    learn_phase(An, Bn, emb, 1.0f);

    float cos_d = cos_to_dir(Ad, Bd, emb + 0, dir02);
    float cos_n = cos_to_dir(An, Bn, emb + 0, dir02);
    printf("after theme switch 0->1 ==> 0->2, alignment with the NEW direction:\n");
    printf("  decay=0.9 (forgetting): cos(δ, dir02) = %.3f\n", cos_d);
    printf("  decay=1.0 (no forget):  cos(δ, dir02) = %.3f\n", cos_n);

    int pass = (cos_d > cos_n + 0.02f);   /* decay must track the new theme better */
    printf(pass ? "\n=== DELTA-DECAY PASS (forgetting tracks recent themes; δ already self-bounds) ===\n"
                : "\n=== DELTA-DECAY FAIL ===\n");
    return pass ? 0 : 1;
}
