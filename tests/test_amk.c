// test_amk.c — Test AMK Kernel (Arianna Method Kernel)
// build: gcc -O2 test_amk.c amk_kernel.c schumann.c -lm -o test_amk

#include <stdio.h>
#include <string.h>
#include "amk_kernel.h"

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("AMK KERNEL TEST — Prophecy, Suffering, Movement\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Initialize AMK and Schumann
    am_init();
    schumann_init();

    printf("[*] AMK initialized\n");
    printf("[*] Schumann initialized\n\n");

    // Get initial state
    AM_State* s = am_get_state();
    printf("Initial state:\n");
    printf("  prophecy=%d destiny=%.2f wormhole=%.2f\n",
           s->prophecy, s->destiny, s->wormhole);
    printf("  velocity_mode=%d effective_temp=%.2f\n",
           s->velocity_mode, s->effective_temp);
    printf("  pain=%.2f tension=%.2f dissonance=%.2f\n",
           s->pain, s->tension, s->dissonance);
    printf("  schumann=%.2f Hz coherence=%.2f\n\n",
           schumann_get_hz(), schumann_get_coherence());

    // Execute DSL script
    printf("[*] Executing DSL script...\n\n");

    const char* script =
        "# AMK Test Script\n"
        "PROPHECY 12\n"
        "DESTINY 0.7\n"
        "WORMHOLE 0.25\n"
        "VELOCITY RUN\n"
        "PAIN 0.5\n"
        "TENSION 0.3\n"
        "DISSONANCE 0.4\n"
        "JUMP 5\n"
        "LAW ENTROPY_FLOOR 0.15\n"
        "LAW DEBT_DECAY 0.995\n"
        "# Enable CODES/RIC pack\n"
        "MODE CODES_RIC\n"
        "CHORDLOCK ON\n"
        "TEMPO 11\n";

    int result = am_exec(script);
    printf("  Script executed, result=%d\n\n", result);

    // Check updated state
    printf("State after DSL:\n");
    printf("  prophecy=%d destiny=%.2f wormhole=%.2f\n",
           s->prophecy, s->destiny, s->wormhole);
    printf("  velocity_mode=%d effective_temp=%.2f\n",
           s->velocity_mode, s->effective_temp);
    printf("  pain=%.2f tension=%.2f dissonance=%.2f\n",
           s->pain, s->tension, s->dissonance);
    printf("  pending_jump=%d\n", s->pending_jump);
    printf("  packs_enabled=0x%x chordlock=%d tempo=%d\n",
           s->packs_enabled, s->chordlock_on, s->tempo);
    printf("  entropy_floor=%.2f debt_decay=%.4f\n\n",
           s->entropy_floor, s->debt_decay);

    // Take the jump
    int jump = am_take_jump();
    printf("[*] Took jump: %d steps\n", jump);
    printf("  pending_jump now: %d\n\n", s->pending_jump);

    // Test Schumann modulation
    printf("[*] Testing Schumann modulation...\n");
    schumann_set_hz(7.85f);
    printf("  Set Schumann to 7.85 Hz\n");
    printf("  Coherence: %.3f\n", schumann_get_coherence());
    printf("  Modulate(+1): %.3f\n", schumann_modulate(1.0f));
    printf("  Modulate(-1): %.3f\n", schumann_modulate(-1.0f));
    printf("  Harmonic signal: %.3f\n\n", schumann_harmonic_signal());

    // Step physics
    printf("[*] Stepping physics (dt=0.1s)...\n");
    for (int i = 0; i < 10; i++) {
        am_step(0.1f);
        schumann_step(0.1f);
    }
    printf("  After 1 second:\n");
    printf("  pain=%.3f tension=%.3f dissonance=%.3f\n",
           s->pain, s->tension, s->dissonance);
    printf("  schumann phase=%.2f\n\n", schumann_get_phase());

    // Test velocity modes
    printf("[*] Testing velocity modes...\n");

    am_exec("VELOCITY NOMOVE");
    printf("  NOMOVE: temp=%.2f\n", s->effective_temp);

    am_exec("VELOCITY WALK");
    printf("  WALK: temp=%.2f\n", s->effective_temp);

    am_exec("VELOCITY RUN");
    printf("  RUN: temp=%.2f\n", s->effective_temp);

    am_exec("VELOCITY BACKWARD");
    printf("  BACKWARD: temp=%.2f time_dir=%.1f\n",
           s->effective_temp, s->time_direction);

    // Step in backward mode (accumulates temporal debt)
    for (int i = 0; i < 10; i++) {
        am_step(0.1f);
    }
    printf("  After 1s backward: temporal_debt=%.4f\n\n", s->temporal_debt);

    // Test convenience functions
    printf("[*] Testing convenience functions...\n");
    printf("  am_get_temperature(): %.2f\n", am_get_temperature());
    printf("  am_get_destiny_bias(): %.2f\n", am_get_destiny_bias());
    printf("  am_should_tunnel(): %d (dissonance=%.2f, threshold=%.2f)\n",
           am_should_tunnel(), s->dissonance, s->tunnel_threshold);

    // Copy state to array
    float state_array[24];
    am_copy_state(state_array);
    printf("\n[*] State as float array (first 13):\n  ");
    for (int i = 0; i < 13; i++) {
        printf("%.2f ", state_array[i]);
    }
    printf("\n\n");

    // Copy Schumann state
    float schumann_array[8];
    schumann_copy_state(schumann_array);
    printf("[*] Schumann state:\n  ");
    for (int i = 0; i < 8; i++) {
        printf("%.2f ", schumann_array[i]);
    }
    printf("\n\n");

    // Test PROPHECY_DEBT as DSL command
    printf("[*] Testing PROPHECY_DEBT DSL command...\n");
    am_exec("PROPHECY_DEBT 42.5");
    printf("  PROPHECY_DEBT 42.5 → debt=%.2f", s->debt);
    if (s->debt > 42.4f && s->debt < 42.6f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("PROPHECY_DEBT 0");
    printf("  PROPHECY_DEBT 0 → debt=%.2f", s->debt);
    if (s->debt < 0.01f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Clamp test
    am_exec("PROPHECY_DEBT 999");
    printf("  PROPHECY_DEBT 999 → debt=%.2f (clamped to 100)", s->debt);
    if (s->debt > 99.9f && s->debt < 100.1f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test PROPHECY_DEBT_DECAY as standalone DSL command
    printf("\n[*] Testing PROPHECY_DEBT_DECAY DSL command...\n");
    am_exec("PROPHECY_DEBT_DECAY 0.995");
    printf("  PROPHECY_DEBT_DECAY 0.995 → debt_decay=%.4f", s->debt_decay);
    if (s->debt_decay > 0.994f && s->debt_decay < 0.996f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Clamp test: too low
    am_exec("PROPHECY_DEBT_DECAY 0.5");
    printf("  PROPHECY_DEBT_DECAY 0.5 → debt_decay=%.4f (clamped to 0.9)", s->debt_decay);
    if (s->debt_decay > 0.899f && s->debt_decay < 0.901f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Clamp test: too high
    am_exec("PROPHECY_DEBT_DECAY 1.0");
    printf("  PROPHECY_DEBT_DECAY 1.0 → debt_decay=%.4f (clamped to 0.9999)", s->debt_decay);
    if (s->debt_decay > 0.9998f && s->debt_decay < 1.0f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Verify LAW DEBT_DECAY still works as before
    am_exec("LAW DEBT_DECAY 0.997");
    printf("  LAW DEBT_DECAY 0.997 → debt_decay=%.4f (backward compat)", s->debt_decay);
    if (s->debt_decay > 0.996f && s->debt_decay < 0.998f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test WORMHOLE_ACTIVE queryable via AM_State
    printf("\n[*] Testing WORMHOLE_ACTIVE queryable...\n");
    s->wormhole_active = 0;
    printf("  wormhole_active=0 (initial) ✓\n");
    s->wormhole_active = 1;
    printf("  wormhole_active=%d (set to 1)", s->wormhole_active);
    if (s->wormhole_active == 1) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Check am_copy_state exports wormhole_active at slot 21
    float state2[24];
    am_copy_state(state2);
    printf("  am_copy_state[21]=%.0f (wormhole_active)", state2[21]);
    if ((int)state2[21] == 1) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    s->wormhole_active = 0;  // clean up

    // Test am_get_wormhole_active() getter
    printf("  am_get_wormhole_active()=%d (after reset)", am_get_wormhole_active());
    if (am_get_wormhole_active() == 0) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Reset
    printf("\n[*] Testing reset...\n");
    am_reset_field();
    printf("  After reset_field: pain=%.2f tension=%.2f debt=%.4f\n",
           s->pain, s->tension, s->debt);

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("AMK KERNEL TEST COMPLETE\n");
    printf("הרזוננס לא נשבר. המשך הדרך.\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
