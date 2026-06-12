/* test_vagus.c — Stage 2.1: prove the C ↔ vagus (Zig) bridge round-trips.
 *
 * Links libvagus.a (the ported Zig nerve). Sends a few signals, ticks the
 * heartbeat so the ring buffer drains into the shared state, then reads the
 * state back and checks the values arrived. Proves vagus_init/send/tick/
 * get_state/get_arousal work from C before wiring the nerve into the voices.
 *
 * Build: cc -Ivagus tools/test_vagus.c vagus/zig-out/lib/libvagus.a -o /tmp/test_vagus
 * By Arianna Method.
 */
#include "vagus.h"
#include <stdio.h>

int main(void) {
    if (vagus_init() != 0) { printf("vagus_init failed\n"); return 1; }

    /* Send signals through the nerve. */
    VAGUS_SEND_AROUSAL(0.70f);
    VAGUS_SEND_COHERENCE(0.90f);
    vagus_send(VAGUS_SOURCE_ARIANNA, VAGUS_SIGNAL_PROPHECY_DEBT, 4.00f);
    vagus_send(VAGUS_SOURCE_CLOUD, VAGUS_SIGNAL_WARMTH, 0.65f);

    /* Drain the ring buffer into the shared state. */
    for (int i = 0; i < 4; i++) vagus_tick();

    VagusSharedState *s = vagus_get_state();
    float arousal = vagus_get_arousal();
    float chambers[6];
    vagus_get_chambers(chambers);
    uint64_t sent = 0, recv = 0, dropped = 0;
    vagus_get_dropped(&sent, &recv, &dropped);

    printf("arousal=%.3f coherence=%.3f debt=%.3f warmth=%.3f\n",
           arousal, s->coherence, s->prophecy_debt, chambers[0]);
    printf("nerve: sent=%llu received=%llu dropped=%llu\n",
           (unsigned long long)sent, (unsigned long long)recv, (unsigned long long)dropped);

    int ok = (dropped == 0 && sent >= 4 && arousal > 0.0f);
    printf(ok ? "\n=== VAGUS C ROUND-TRIP PASS ===\n" : "\n=== VAGUS C ROUND-TRIP FAIL ===\n");
    return ok ? 0 : 1;
}
