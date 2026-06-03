/* test_cooc_consolidate.c — B2-A unit proof for autumn cooc consolidation.
 *
 * Builds a deterministic co-occurrence field (two strong edges, three weak
 * "noise" edges), then proves:
 *   1. the autumn gate is a no-op outside autumn (ablation / B1-identity);
 *   2. am_cooc_consolidate reinforces strong edges, decays weak ones, and
 *      prunes everything that falls under the floor (forgetting);
 *   3. the gate fires in deep autumn and prunes a re-added weak edge.
 *
 * Link: cc test_cooc_consolidate.c libaml.a libnotorch.a -framework Accelerate -lm
 *
 * By Arianna Method.
 */
#include "ariannamethod.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", (msg)); fails++; } \
    else         { printf("ok:   %s\n", (msg)); } \
} while (0)

int main(void) {
    am_init();

    /* Deterministic cooc: two strong edges, three weak (noise) edges. */
    am_cooc_update(0, 1, 5.00f);
    am_cooc_update(2, 3, 4.00f);
    am_cooc_update(4, 5, 0.25f);
    am_cooc_update(6, 7, 0.20f);
    am_cooc_update(8, 9, 0.15f);

    /* Ablation: outside autumn (default season SPRING) the gate does nothing. */
    CHECK(am_cooc_consolidate_autumn() == -1, "gate no-op outside autumn");
    CHECK(am_cooc_count() == 5, "cooc untouched outside autumn (5 edges)");

    float m0, x0; am_cooc_stats(&m0, &x0);
    int before = am_cooc_count();

    /* Direct consolidate: median = 0.25, reinforce >=median, prune < 0.30.
     * {5,4,0.25}*1.1 = {5.5,4.4,0.275}; {0.20,0.15}*0.9 = {0.18,0.135}.
     * prune < 0.30 drops 0.275, 0.18, 0.135 -> 2 survive. */
    int pruned = am_cooc_consolidate(0.10f, AM_COOC_AUTUMN_PRUNE);
    int after = am_cooc_count();
    float m1, x1; am_cooc_stats(&m1, &x1);

    printf("before=%d after=%d pruned=%d  mean %.3f->%.3f  max %.3f->%.3f\n",
           before, after, pruned, m0, m1, x0, x1);

    CHECK(before == 5,               "started with 5 edges");
    CHECK(pruned == 3,               "pruned 3 weak edges");
    CHECK(after == 2,                "2 strong edges survive");
    CHECK(x1 > x0,                   "max edge reinforced (5.0 -> 5.5)");
    CHECK(m1 > m0,                   "mean weight rose (noise forgotten)");
    CHECK(fabsf(x1 - 5.5f) < 1e-3f,  "max == 5.5 exactly");

    /* Gate end-to-end: force deep autumn, gate must fire and prune. */
    am_cooc_update(10, 11, 0.10f);   /* re-add a prunable edge */
    AM_State* s = am_get_state();
    s->season = AM_SEASON_AUTUMN;
    s->autumn_energy = 0.8f;
    int gp = am_cooc_consolidate_autumn();
    printf("autumn gate fired: pruned=%d edges=%d\n", gp, am_cooc_count());
    CHECK(gp >= 0, "gate fires in deep autumn");
    CHECK(gp >= 1, "gate prunes the re-added weak edge");

    printf(fails ? "\n=== UNIT FAIL (%d) ===\n" : "\n=== UNIT PASS ===\n", fails);
    return fails ? 1 : 0;
}
