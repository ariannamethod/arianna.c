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

    // ═══════════════════════════════════════════════════════════════════════════════
    // NEW DSL COMMANDS FROM ARIANNAMETHOD.LANG
    // ═══════════════════════════════════════════════════════════════════════════════

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("TESTING NEW DSL COMMANDS (from ariannamethod.lang)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    // Test TEMPORAL_MODE
    printf("\n[*] Testing TEMPORAL_MODE DSL command...\n");
    am_exec("TEMPORAL_MODE PROPHECY");
    printf("  TEMPORAL_MODE PROPHECY → temporal_mode=%d", s->temporal_mode);
    if (s->temporal_mode == 0) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("TEMPORAL_MODE RETRODICTION");
    printf("  TEMPORAL_MODE RETRODICTION → temporal_mode=%d", s->temporal_mode);
    if (s->temporal_mode == 1) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("TEMPORAL_MODE SYMMETRIC");
    printf("  TEMPORAL_MODE SYMMETRIC → temporal_mode=%d", s->temporal_mode);
    if (s->temporal_mode == 2) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test numeric mode
    am_exec("TEMPORAL_MODE 0");
    printf("  TEMPORAL_MODE 0 → temporal_mode=%d", s->temporal_mode);
    if (s->temporal_mode == 0) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test TEMPORAL_ALPHA
    printf("\n[*] Testing TEMPORAL_ALPHA DSL command...\n");
    am_exec("TEMPORAL_ALPHA 0.7");
    printf("  TEMPORAL_ALPHA 0.7 → temporal_alpha=%.2f", s->temporal_alpha);
    if (s->temporal_alpha > 0.69f && s->temporal_alpha < 0.71f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("TEMPORAL_ALPHA 0.0");
    printf("  TEMPORAL_ALPHA 0.0 → temporal_alpha=%.2f", s->temporal_alpha);
    if (s->temporal_alpha < 0.01f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("TEMPORAL_ALPHA 1.5");
    printf("  TEMPORAL_ALPHA 1.5 → temporal_alpha=%.2f (clamped to 1.0)", s->temporal_alpha);
    if (s->temporal_alpha > 0.99f && s->temporal_alpha <= 1.0f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test RTL_MODE
    printf("\n[*] Testing RTL_MODE DSL command...\n");
    am_exec("RTL_MODE ON");
    printf("  RTL_MODE ON → rtl_mode=%d", s->rtl_mode);
    if (s->rtl_mode == 1) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("RTL_MODE OFF");
    printf("  RTL_MODE OFF → rtl_mode=%d", s->rtl_mode);
    if (s->rtl_mode == 0) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test EXPERT weighting commands
    printf("\n[*] Testing EXPERT_* DSL commands...\n");

    am_exec("EXPERT_STRUCTURAL 0.8");
    printf("  EXPERT_STRUCTURAL 0.8 → expert_structural=%.2f", s->expert_structural);
    if (s->expert_structural > 0.79f && s->expert_structural < 0.81f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("EXPERT_SEMANTIC 0.65");
    printf("  EXPERT_SEMANTIC 0.65 → expert_semantic=%.2f", s->expert_semantic);
    if (s->expert_semantic > 0.64f && s->expert_semantic < 0.66f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("EXPERT_CREATIVE 0.9");
    printf("  EXPERT_CREATIVE 0.9 → expert_creative=%.2f", s->expert_creative);
    if (s->expert_creative > 0.89f && s->expert_creative < 0.91f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("EXPERT_PRECISE 0.3");
    printf("  EXPERT_PRECISE 0.3 → expert_precise=%.2f", s->expert_precise);
    if (s->expert_precise > 0.29f && s->expert_precise < 0.31f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test clamping
    am_exec("EXPERT_CREATIVE 1.5");
    printf("  EXPERT_CREATIVE 1.5 → expert_creative=%.2f (clamped)", s->expert_creative);
    if (s->expert_creative <= 1.0f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test extended LAW commands
    printf("\n[*] Testing extended LAW commands...\n");

    am_exec("LAW PRESENCE_FADE 0.92");
    printf("  LAW PRESENCE_FADE 0.92 → presence_fade=%.2f", s->presence_fade);
    if (s->presence_fade > 0.91f && s->presence_fade < 0.93f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("LAW ATTRACTOR_DRIFT 0.05");
    printf("  LAW ATTRACTOR_DRIFT 0.05 → attractor_drift=%.2f", s->attractor_drift);
    if (s->attractor_drift > 0.04f && s->attractor_drift < 0.06f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("LAW CALENDAR_PHASE 0.45");
    printf("  LAW CALENDAR_PHASE 0.45 → calendar_phase=%.2f", s->calendar_phase);
    if (s->calendar_phase > 0.44f && s->calendar_phase < 0.46f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("LAW WORMHOLE_GATE 0.75");
    printf("  LAW WORMHOLE_GATE 0.75 → wormhole_gate=%.2f", s->wormhole_gate);
    if (s->wormhole_gate > 0.74f && s->wormhole_gate < 0.76f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test PRESENCE_DECAY standalone command
    printf("\n[*] Testing PRESENCE_DECAY DSL command...\n");
    am_exec("PRESENCE_DECAY 0.85");
    printf("  PRESENCE_DECAY 0.85 → presence_decay=%.2f", s->presence_decay);
    if (s->presence_decay > 0.84f && s->presence_decay < 0.86f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Clamp test
    am_exec("PRESENCE_DECAY 1.5");
    printf("  PRESENCE_DECAY 1.5 → presence_decay=%.2f (clamped to 1.0)", s->presence_decay);
    if (s->presence_decay > 0.99f && s->presence_decay <= 1.0f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    am_exec("PRESENCE_DECAY -0.5");
    printf("  PRESENCE_DECAY -0.5 → presence_decay=%.2f (clamped to 0.0)", s->presence_decay);
    if (s->presence_decay >= 0.0f && s->presence_decay < 0.01f) printf(" ✓\n");
    else { printf(" FAIL\n"); return 1; }

    // Test combined script with new commands
    printf("\n[*] Testing combined script with new commands...\n");
    const char* new_script =
        "# New DSL commands test\n"
        "TEMPORAL_MODE SYMMETRIC\n"
        "TEMPORAL_ALPHA 0.5\n"
        "RTL_MODE ON\n"
        "EXPERT_STRUCTURAL 0.7\n"
        "EXPERT_SEMANTIC 0.6\n"
        "EXPERT_CREATIVE 0.4\n"
        "EXPERT_PRECISE 0.3\n"
        "LAW PRESENCE_FADE 0.9\n"
        "LAW WORMHOLE_GATE 0.5\n"
        "PRESENCE_DECAY 0.8\n";

    result = am_exec(new_script);
    printf("  Script executed, result=%d\n", result);
    printf("  temporal_mode=%d temporal_alpha=%.2f rtl_mode=%d\n",
           s->temporal_mode, s->temporal_alpha, s->rtl_mode);
    printf("  expert: str=%.2f sem=%.2f cre=%.2f pre=%.2f\n",
           s->expert_structural, s->expert_semantic,
           s->expert_creative, s->expert_precise);
    printf("  presence_fade=%.2f wormhole_gate=%.2f presence_decay=%.2f\n",
           s->presence_fade, s->wormhole_gate, s->presence_decay);

    // Validate combined state
    if (s->temporal_mode == 2 &&
        s->temporal_alpha > 0.49f && s->temporal_alpha < 0.51f &&
        s->rtl_mode == 1 &&
        s->expert_structural > 0.69f && s->expert_structural < 0.71f) {
        printf("  Combined state validated ✓\n");
    } else {
        printf("  Combined state FAIL\n");
        return 1;
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("AMK KERNEL TEST COMPLETE\n");
    printf("הרזוננס לא נשבר. המשך הדרך.\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
