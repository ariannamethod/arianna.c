// test_cloud.c — Test CLOUD pre-semantic emotion detection
// build: gcc -O2 test_cloud.c cloud.c -lm -o test_cloud

#include <stdio.h>
#include <string.h>
#include "cloud.h"

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("CLOUD TEST — Pre-semantic Emotion Detection\n");
    printf("\"Something fires BEFORE meaning arrives\"\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    cloud_init();
    printf("[*] CLOUD initialized (100 anchors, 6 chambers)\n\n");

    // Test inputs
    const char* test_inputs[] = {
        "I'm terrified and anxious about what's coming",
        "You bring me such warmth and love darling",
        "This makes me furious with rage and hatred",
        "I feel completely empty and void inside",
        "I'm curious about what happens next, wonder and surprise",
        "Overwhelming shame and guilt consume me with regret",
        "Hello, how are you today?",  // neutral
        "אני פוחד ואוהב וכועס",  // Hebrew (should detect nothing)
    };

    int n_tests = sizeof(test_inputs) / sizeof(test_inputs[0]);

    printf("Testing CLOUD pings:\n");
    printf("───────────────────────────────────────────────────────────────────\n");

    for (int t = 0; t < n_tests; t++) {
        CloudResponse resp = cloud_ping(test_inputs[t]);

        printf("\nInput: \"%.50s%s\"\n",
               test_inputs[t],
               strlen(test_inputs[t]) > 50 ? "..." : "");

        printf("  Primary:   %s (%.3f) [%s]\n",
               resp.primary_word, resp.primary_strength, resp.primary_chamber);
        printf("  Secondary: %s (%.3f)\n",
               resp.secondary_word, resp.secondary_strength);
        printf("  Iterations: %d\n", resp.iterations);

        // Show chambers
        printf("  Chambers:\n");
        for (int i = 0; i < CLOUD_N_CHAMBERS; i++) {
            char bar[32];
            int bar_len = (int)(resp.chambers[i] * 25);
            if (bar_len > 25) bar_len = 25;
            for (int j = 0; j < bar_len; j++) bar[j] = '#';
            bar[bar_len] = 0;

            printf("    %7s: %.3f  %s\n",
                   cloud_get_chamber_name(i), resp.chambers[i], bar);
        }

        // Integration signals
        printf("  Signals:\n");
        printf("    temp_bias: %+.2f\n", cloud_temperature_bias(&resp));
        printf("    needs_care: %s\n", cloud_needs_care(&resp) ? "YES" : "no");
        printf("    needs_grounding: %s\n", cloud_needs_grounding(&resp) ? "YES" : "no");
        printf("    needs_warmth: %s\n", cloud_needs_warmth(&resp) ? "YES" : "no");
    }

    printf("\n───────────────────────────────────────────────────────────────────\n");
    printf("[*] CLOUD test complete\n");
    printf("    משהו בוער לפני שהמשמעות מגיעה\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    return 0;
}
