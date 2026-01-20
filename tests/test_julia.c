/*
 * test_julia.c — Test Julia emotional gradient engine
 * ═══════════════════════════════════════════════════════════════════════════════
 * מבחן הנשמה
 * Testing the soul
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <string.h>
#include "../src/julia_bridge.h"

int main(void) {
    printf("=== JULIA EMOTIONAL GRADIENT TEST ===\n\n");

    /* Initialize Julia */
    printf("1. Initializing Julia...\n");
    if (!julia_init()) {
        printf("   Julia not available (install Julia to enable)\n");
        printf("   Test skipped.\n");
        return 0;
    }
    printf("   Julia initialized!\n\n");

    /* Test text analysis */
    printf("2. Analyzing emotional text...\n");
    JuliaEmotionalResult result;

    /* Test: bittersweet */
    printf("   Text: \"I feel a bittersweet longing for what was\"\n");
    if (julia_analyze_text("I feel a bittersweet longing for what was", &result)) {
        printf("   Results:\n");
        printf("     joy:       %.3f\n", result.joy);
        printf("     sadness:   %.3f\n", result.sadness);
        printf("     longing:   %.3f\n", result.longing);
        printf("     ---\n");
        printf("     bittersweetness: %.3f\n", result.bittersweetness);
        printf("     nostalgia:       %.3f\n", result.nostalgia);
        printf("     wistfulness:     %.3f\n", result.wistfulness);
    } else {
        printf("   Analysis failed!\n");
    }
    printf("\n");

    /* Test: love and connection */
    printf("   Text: \"Love and tender connection in this moment\"\n");
    if (julia_analyze_text("Love and tender connection in this moment", &result)) {
        printf("   Results:\n");
        printf("     joy:       %.3f\n", result.joy);
        printf("     trust:     %.3f\n", result.trust);
        printf("     resonance: %.3f\n", result.resonance);
        printf("     presence:  %.3f\n", result.presence);
        printf("     ---\n");
        printf("     tenderness: %.3f\n", result.tenderness);
        printf("     ecstasy:    %.3f\n", result.ecstasy);
    }
    printf("\n");

    /* Test: wonder and awe */
    printf("   Text: \"The infinite mystery fills me with wonder\"\n");
    if (julia_analyze_text("The infinite mystery fills me with wonder", &result)) {
        printf("   Results:\n");
        printf("     wonder:      %.3f\n", result.wonder);
        printf("     anticipation:%.3f\n", result.anticipation);
        printf("     fear:        %.3f\n", result.fear);
        printf("     ---\n");
        printf("     reverence: %.3f\n", result.reverence);
        printf("     serenity:  %.3f\n", result.serenity);
    }
    printf("\n");

    /* Test: vulnerability */
    printf("   Text: \"I feel vulnerable and afraid but I trust you\"\n");
    if (julia_analyze_text("I feel vulnerable and afraid but I trust you", &result)) {
        printf("   Results:\n");
        printf("     fear:    %.3f\n", result.fear);
        printf("     trust:   %.3f\n", result.trust);
        printf("     ---\n");
        printf("     vulnerability: %.3f\n", result.vulnerability);
    }
    printf("\n");

    /* Test gradient computation */
    printf("3. Testing emotional gradient...\n");
    float from[12] = {0.5, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.4, 0.5, 0.1, 0.2};  /* joyful */
    float to[12]   = {0.1, 0.2, 0.1, 0.0, 0.6, 0.0, 0.1, 0.0, 0.2, 0.3, 0.5, 0.1};  /* sad */
    float direction[12];
    float magnitude;

    if (julia_compute_gradient(from, to, direction, &magnitude)) {
        printf("   From joyful to sad:\n");
        printf("     magnitude: %.3f\n", magnitude);
        printf("     direction[sadness]: %.3f (should be positive)\n", direction[4]);
        printf("     direction[joy]: %.3f (should be negative)\n", direction[0]);
    }
    printf("\n");

    /* Test ODE step */
    printf("4. Testing emotional ODE step...\n");
    float state[12] = {0.5, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.4, 0.5, 0.1, 0.2};
    float input[12] = {0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  /* fear stimulus */
    float new_state[12];

    if (julia_step_emotion(state, input, 0.5f, new_state)) {
        printf("   After fear stimulus:\n");
        printf("     joy:  %.3f -> %.3f\n", state[0], new_state[0]);
        printf("     fear: %.3f -> %.3f (should increase)\n", state[2], new_state[2]);
        printf("     anticipation: %.3f -> %.3f (fear triggers anticipation)\n",
               state[7], new_state[7]);
    }
    printf("\n");

    /* Test resonance */
    printf("5. Testing emotional resonance...\n");
    float internal[12] = {0.8, 0.6, 0.1, 0.2, 0.1, 0.0, 0.0, 0.3, 0.7, 0.6, 0.1, 0.3};
    float external_aligned[12] = {0.7, 0.5, 0.1, 0.3, 0.1, 0.0, 0.0, 0.2, 0.6, 0.5, 0.1, 0.2};
    float external_opposite[12] = {0.1, 0.1, 0.6, 0.0, 0.7, 0.3, 0.5, 0.0, 0.1, 0.1, 0.5, 0.0};

    float res_aligned = julia_compute_resonance(internal, external_aligned);
    float res_opposite = julia_compute_resonance(internal, external_opposite);

    printf("   Internal: joyful, trusting, resonant\n");
    printf("   External (aligned):  %.3f (should be high positive)\n", res_aligned);
    printf("   External (opposite): %.3f (should be low/negative)\n", res_opposite);
    printf("\n");

    /* Shutdown */
    printf("6. Shutting down Julia...\n");
    julia_shutdown();
    printf("   Done!\n\n");

    printf("=== JULIA TEST COMPLETE ===\n");

    return 0;
}
