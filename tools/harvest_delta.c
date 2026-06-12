/* harvest_delta.c — B2-B.3: harvest a real low-rank δ from a voice's own cooc.
 *
 * The autumn block in the .aml is the *trigger* for the harvest; the harvest
 * itself is am_cooc_learn_delta, which can be called directly. This tool folds a
 * voice's real co-occurrence sidecar (built from real dialogue) into the
 * per-voice δ sidecar, using that voice's real token embeddings — so the
 * behavioural run has a real, non-zero δ to toggle with LORA_ALPHA.
 *
 * Usage: harvest_delta <gguf> <wte_tensor> <cooc_sidecar> <delta_out> <V> <E> [passes]
 * Link:  cc -Iariannamethod/notorch -Iariannamethod/core harvest_delta.c \
 *           ariannamethod/core/libaml.a ariannamethod/notorch/libnotorch.a \
 *           -framework Accelerate -lm
 *
 * By Arianna Method.
 */
#include "ariannamethod.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <gguf> <wte_tensor> <cooc> <delta_out> <V> <E> [passes]\n", argv[0]);
        return 2;
    }
    const char *gguf = argv[1], *wte_name = argv[2], *cooc = argv[3], *dout = argv[4];
    int V = atoi(argv[5]), E = atoi(argv[6]);
    int passes = (argc > 7) ? atoi(argv[7]) : 50;
    int rank = AM_DELTA_RANK;

    am_init();

    /* real cooc from the voice's own sidecar (built from real dialogue) */
    if (am_cooc_load(cooc) != 0) { fprintf(stderr, "[harvest] no cooc sidecar '%s'\n", cooc); return 1; }
    int edges = am_cooc_count();
    if (edges <= 0) { fprintf(stderr, "[harvest] cooc empty — nothing to harvest\n"); return 1; }

    /* real token embeddings from the voice's GGUF */
    gguf_file *gf = gguf_open(gguf);
    if (!gf) { fprintf(stderr, "[harvest] gguf_open('%s') failed\n", gguf); return 1; }
    int idx = gguf_find_tensor(gf, wte_name);
    if (idx < 0) { fprintf(stderr, "[harvest] tensor '%s' not found\n", wte_name); gguf_close(gf); return 1; }
    float *wte = gguf_dequant(gf, idx);
    if (!wte) { fprintf(stderr, "[harvest] dequant '%s' failed\n", wte_name); gguf_close(gf); return 1; }

    float *A = (float*)calloc((size_t)E * rank, sizeof(float));
    float *B = (float*)calloc((size_t)rank * E, sizeof(float));
    if (!A || !B) { fprintf(stderr, "[harvest] OOM\n"); return 1; }

    /* fold the cooc edges into (A,B) over several passes (cf. repeated autumns) */
    int folded = 0;
    for (int p = 0; p < passes; p++) folded += am_cooc_learn_delta(A, B, wte, V, E, rank);

    double na = 0, nb = 0;
    for (size_t i = 0; i < (size_t)E * rank; i++) na += (double)A[i] * A[i];
    for (size_t i = 0; i < (size_t)rank * E; i++) nb += (double)B[i] * B[i];
    fprintf(stderr, "[harvest] cooc edges=%d, %d edge-passes folded, |A|=%.5f |B|=%.5f\n",
            edges, folded, sqrt(na), sqrt(nb));
    if (na == 0.0 || nb == 0.0) { fprintf(stderr, "[harvest] δ is ZERO — refusing to save\n"); return 1; }

    if (am_delta_save(dout, A, B, E, rank) != 0) { fprintf(stderr, "[harvest] save '%s' failed\n", dout); return 1; }
    fprintf(stderr, "[harvest] saved real δ → %s (rank=%d)\n", dout, rank);

    gguf_close(gf);
    return 0;
}
