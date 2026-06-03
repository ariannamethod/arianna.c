/* tools/jannus_split.h — header-only chain split + temp ladder.
 *
 * Extracted from jannus_run_chain so the test suite can verify the math
 * without booting an inference. Source of truth for:
 *
 *   nb = clamp(NSTEPS * (0.3 + 0.4*debt + 0.1*cal_diss), 1, NSTEPS-1)
 *   nf = NSTEPS - nb
 *   tb = 0.7 + 0.3 * (0.5 + 0.3*cal_diss + 0.2*debt)
 *
 *   forward step i temp = tb * (1 - 0.02*i)   — focus rises with depth
 *   backward step i temp = tb * (1 + 0.05*i)  — exploration rises with reach
 *
 * Anyone changing these constants must update both the chain caller and
 * test_split — diverging copies were the original reason for extracting.
 */
#ifndef JANNUS_SPLIT_H
#define JANNUS_SPLIT_H

typedef struct {
    int   nb;       /* backward count */
    int   nf;       /* forward count == NSTEPS - nb */
    float tb;       /* base temperature */
} JannusSplit;

/* Compute nb / nf / tb from prophecy debt and calendar dissonance. */
static inline JannusSplit jannus_compute_split(float debt, float cal_diss, int nsteps) {
    JannusSplit s;
    int nb = (int)((float)nsteps * (0.3f + 0.4f * debt + 0.1f * cal_diss));
    if (nb < 1)            nb = 1;
    if (nb >= nsteps)      nb = nsteps - 1;
    s.nb = nb;
    s.nf = nsteps - nb;
    s.tb = 0.7f + 0.3f * (0.5f + 0.3f * cal_diss + 0.2f * debt);
    return s;
}

/* Per-step temperature for forward branch. i ∈ [0, nf). Decreasing in i. */
static inline float jannus_temp_forward(float tb, int i)  { return tb * (1.0f - 0.02f * (float)i); }
/* Per-step temperature for backward branch. i ∈ [0, nb). Increasing in i. */
static inline float jannus_temp_backward(float tb, int i) { return tb * (1.0f + 0.05f * (float)i); }

#endif /* JANNUS_SPLIT_H */
