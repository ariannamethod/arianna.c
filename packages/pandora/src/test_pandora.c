// test_pandora.c — Test Pandora package
// ═══════════════════════════════════════════════════════════════════════════════

#include "pandora.h"
#include <stdio.h>
#include <string.h>

// Mock Arianna encoder
static int mock_arianna_encode(const char* word) {
    // Simple hash to mock token ID
    int h = 0;
    while (*word) {
        h = h * 31 + *word++;
    }
    return h & 0x3F;  // 64-token mock vocab
}

// Mock GPT2 decoder (use global vocab for testing)
static GPT2Vocab test_vocab;

static const char* mock_brain_decode(int token_id) {
    return gpt2_vocab_decode(&test_vocab, token_id);
}

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("PANDORA PACKAGE TEST\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    PandoraBox box;
    int passed = 0;
    int total = 0;

    // Test 1: Basic init
    printf("Test 1: Basic init... ");
    total++;
    pandora_init(&box);
    if (box.mode == PANDORA_MODE_AUTO && box.injection_strength == 0.2f) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 2: N-gram extraction
    printf("Test 2: N-gram extraction... ");
    total++;
    int tokens[] = {1, 2, 3, 4, 5, 1, 2, 3};  // Has repeated bigrams
    pandora_extract(&box, tokens, 8, 1, 2);
    if (box.n_ngrams > 0) {
        printf("PASS (%d n-grams)\n", box.n_ngrams);
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 3: Repeated extraction boosts weight
    printf("Test 3: Weight boosting... ");
    total++;
    float initial_weight = box.ngrams[0].weight;
    pandora_extract(&box, tokens, 8, 1, 2);
    if (box.ngrams[0].weight > initial_weight) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 4: SARTRE check - low coherence activates
    printf("Test 4: SARTRE low coherence... ");
    total++;
    if (pandora_check_sartre(&box, 0.2f, 0.3f, 0) == 1) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 5: SARTRE check - high sacred deactivates
    printf("Test 5: SARTRE high sacred... ");
    total++;
    if (pandora_check_sartre(&box, 0.5f, 0.8f, 0) == 0) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 6: SARTRE check - CRISIS deactivates
    printf("Test 6: SARTRE CRISIS... ");
    total++;
    if (pandora_check_sartre(&box, 0.5f, 0.3f, 1) == 0) {  // 1 = CRISIS
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 7: SARTRE check - EMERGENCE activates
    printf("Test 7: SARTRE EMERGENCE... ");
    total++;
    if (pandora_check_sartre(&box, 0.5f, 0.3f, 3) == 1) {  // 3 = EMERGENCE
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 8: Mode switching
    printf("Test 8: Mode switching... ");
    total++;
    pandora_set_mode(&box, PANDORA_MODE_OFF);
    if (!pandora_is_active(&box)) {
        pandora_set_mode(&box, PANDORA_MODE_FORCED);
        if (pandora_is_active(&box)) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL (forced mode)\n");
        }
    } else {
        printf("FAIL (off mode)\n");
    }

    // Test 9: Decay
    printf("Test 9: Decay... ");
    total++;
    pandora_set_mode(&box, PANDORA_MODE_AUTO);
    int before = box.n_ngrams;
    for (int i = 0; i < 100; i++) {
        pandora_decay(&box, 0.9f);
    }
    if (box.n_ngrams < before) {
        printf("PASS (%d -> %d)\n", before, box.n_ngrams);
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Test 10: Save/Load
    printf("Test 10: Save/Load... ");
    total++;
    pandora_clear(&box);
    int test_tokens[] = {10, 20, 30, 40, 50};
    pandora_extract(&box, test_tokens, 5, 2, 3);
    int ngrams_before = box.n_ngrams;

    if (pandora_save(&box, "/tmp/pandora_test.bin") == 0) {
        PandoraBox box2;
        pandora_init(&box2);
        if (pandora_load(&box2, "/tmp/pandora_test.bin") == 0) {
            if (box2.n_ngrams == ngrams_before) {
                printf("PASS\n");
                passed++;
            } else {
                printf("FAIL (n-grams mismatch: %d vs %d)\n", box2.n_ngrams, ngrams_before);
            }
        } else {
            printf("FAIL (load)\n");
        }
    } else {
        printf("FAIL (save)\n");
    }

    // Test 11: Stats
    printf("Test 11: Stats... ");
    total++;
    PandoraStats stats = pandora_get_stats(&box);
    if (stats.total_ngrams == box.n_ngrams) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
    }

    // Summary
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS: %d/%d passed\n", passed, total);
    printf("═══════════════════════════════════════════════════════════════════\n");

    return passed == total ? 0 : 1;
}
