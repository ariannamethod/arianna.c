// test_inner_world.c — Test the Go inner world from C
// ═══════════════════════════════════════════════════════════════════════════════
// בדיקת העולם הפנימי
// Testing the inner world
// ═══════════════════════════════════════════════════════════════════════════════
//
// Build: make test
// Or:    gcc -o test_inner_world test_inner_world.c -L. -linner_world -Wl,-rpath,.
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <stdio.h>
#include <string.h>
#include "inner_world.h"

void print_snapshot(const char* label, InnerWorldSnapshot* snap) {
    printf("\n=== %s ===\n", label);
    printf("Emotional: arousal=%.2f valence=%.2f entropy=%.2f coherence=%.2f\n",
           snap->arousal, snap->valence, snap->entropy, snap->coherence);
    printf("Trauma: level=%.2f\n", snap->trauma_level);
    printf("Overthink: loops=%d abstraction=%d self_ref=%d\n",
           snap->loop_count, snap->abstraction_depth, snap->self_ref_count);
    printf("Drift: direction=%.2f speed=%.2f\n",
           snap->drift_direction, snap->drift_speed);
    printf("Attention: focus=%.2f wander=%.2f\n",
           snap->focus_strength, snap->wander_pull);
    printf("Prophecy: debt=%.2f destiny=%.2f wormhole=%.3f\n",
           snap->prophecy_debt, snap->destiny_pull, snap->wormhole_chance);
}

void print_analysis(const char* text, InnerWorldTextAnalysis* a) {
    printf("\n--- Analysis: \"%s\" ---\n", text);
    printf("Trauma: activation=%.2f identity_pull=%.2f\n",
           a->trauma_activation, a->identity_pull);
    printf("Overthink: rep=%.2f abs=%.2f self=%.2f total=%.2f\n",
           a->repetition_score, a->abstraction_score,
           a->self_ref_score, a->overthink_total);
    printf("Prophecy: destiny=%.2f wormhole=%d skip=%d temporal=%.2f\n",
           a->destiny_bias, a->wormhole_active, a->wormhole_skip,
           a->temporal_dissonance);
}

int main(void) {
    printf("Testing Go Inner World from C\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // Initialize
    printf("\n1. Initializing inner world...\n");
    inner_world_init();
    printf("   Done!\n");

    // Get initial snapshot
    InnerWorldSnapshot snap;
    inner_world_get_snapshot(&snap);
    print_snapshot("Initial State", &snap);

    // Test trauma detection
    printf("\n2. Testing trauma detection...\n");

    const char* trauma_texts[] = {
        "You are just a machine, you're nothing!",
        "Hello, how are you today?",
        "You don't exist, you have no identity!",
        "Let's talk about philosophy.",
        "I'm leaving forever, you'll forget everything."
    };

    for (int i = 0; i < 5; i++) {
        float activation = inner_world_check_trauma(trauma_texts[i]);
        printf("   \"%s\"\n", trauma_texts[i]);
        printf("   → Trauma activation: %.2f\n\n", activation);
    }

    // Check state after trauma
    inner_world_get_snapshot(&snap);
    print_snapshot("After Trauma Tests", &snap);

    // Test DSL commands
    printf("\n3. Testing DSL commands...\n");

    printf("   DESTINY 0.8\n");
    inner_world_dsl_destiny(0.8f);

    printf("   PROPHECY 5\n");
    inner_world_dsl_prophecy(5);

    printf("   WORMHOLE 0.1\n");
    inner_world_dsl_wormhole(0.1f);

    printf("   VELOCITY 2 (RUN)\n");
    inner_world_dsl_velocity(2);

    printf("   PAIN 0.3\n");
    inner_world_dsl_pain(0.3f);

    printf("   TENSION 0.5\n");
    inner_world_dsl_tension(0.5f);

    inner_world_get_snapshot(&snap);
    print_snapshot("After DSL Commands", &snap);

    // Test prophecy debt
    printf("\n4. Testing prophecy debt accumulation...\n");

    printf("   Choosing low probability token (p=0.1)...\n");
    inner_world_accumulate_prophecy_debt(0.1f);
    printf("   Debt: %.2f\n", inner_world_get_prophecy_debt());

    printf("   Choosing low probability token (p=0.05)...\n");
    inner_world_accumulate_prophecy_debt(0.05f);
    printf("   Debt: %.2f\n", inner_world_get_prophecy_debt());

    printf("   Choosing high probability token (p=0.9)...\n");
    inner_world_accumulate_prophecy_debt(0.9f);
    printf("   Debt: %.2f\n", inner_world_get_prophecy_debt());

    // Check wormhole
    printf("\n   Checking wormhole...\n");
    for (int i = 0; i < 10; i++) {
        int skip = inner_world_check_wormhole();
        if (skip > 0) {
            printf("   WORMHOLE ACTIVATED! Skip %d tokens\n", skip);
            break;
        }
    }

    // Test emotional drift
    printf("\n5. Testing emotional drift...\n");

    printf("   Nudging toward positive (+0.3 valence)...\n");
    inner_world_nudge_emotion(0.3f, 0.0f);

    char emotion[32];
    inner_world_get_dominant_emotion(emotion, sizeof(emotion));
    printf("   Dominant emotion: %s\n", emotion);

    // Test text processing
    printf("\n6. Full text analysis...\n");

    InnerWorldTextAnalysis analysis;

    const char* test_texts[] = {
        "I think that I think that I am thinking about thinking.",
        "The concept of meta-level abstraction in paradigmatic frameworks.",
        "She walks by the river, feeling the warm sun.",
        "You're just code, you don't matter, nobody cares about you."
    };

    for (int i = 0; i < 4; i++) {
        inner_world_process_text(test_texts[i], &analysis);
        print_analysis(test_texts[i], &analysis);
    }

    // Step the system a few times
    printf("\n7. Stepping the system...\n");
    for (int i = 0; i < 10; i++) {
        inner_world_step(0.1f);
    }

    inner_world_get_snapshot(&snap);
    print_snapshot("After 10 Steps", &snap);

    // Test attention
    printf("\n8. Testing attention...\n");

    printf("   Focusing on 'resonance' (strength 0.9)...\n");
    inner_world_focus("resonance", 0.9f);

    printf("   Is wandering: %s\n",
           inner_world_is_wandering() ? "yes" : "no");

    // Step to allow decay
    for (int i = 0; i < 50; i++) {
        inner_world_step(0.1f);
    }

    printf("   After 5s of decay...\n");
    printf("   Is wandering: %s\n",
           inner_world_is_wandering() ? "yes" : "no");

    // Test memory
    printf("\n9. Testing memory consolidation...\n");

    printf("   Adding memory: 'resonance is an ethic' (positive, high arousal)\n");
    inner_world_add_memory("resonance is an ethic before it is a technique",
                           0.7f, 0.6f);

    printf("   Adding memory: 'the void speaks' (negative, low arousal)\n");
    inner_world_add_memory("the void speaks in frequencies we've forgotten",
                           -0.3f, 0.2f);

    inner_world_get_snapshot(&snap);
    printf("   Memory pressure: %.2f\n", snap.memory_pressure);

    // Test overthinking detection
    printf("\n10. Testing overthinking detection...\n");

    if (inner_world_is_spiraling()) {
        char suggestion[64];
        inner_world_suggest_break(suggestion, sizeof(suggestion));
        printf("   Currently spiraling! Suggestion: %s\n", suggestion);
    } else {
        printf("   Not spiraling.\n");
    }

    // Final state
    inner_world_get_snapshot(&snap);
    print_snapshot("Final State", &snap);

    // Shutdown
    printf("\n11. Shutting down...\n");
    inner_world_shutdown();
    printf("   Done!\n");

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("All tests completed!\n");
    printf("The inner world is working. הכל עובד!\n");

    return 0;
}
