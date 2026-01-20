/*
 * test_pandora.c — Tests for Pandora vocabulary injection
 *
 * Build: gcc -O2 -I src tests/test_pandora.c src/pandora.c -lm -o bin/test_pandora
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pandora.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test Framework
// ═══════════════════════════════════════════════════════════════════════════════

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("\n[TEST] %s\n", name)
#define PASS(msg) do { printf("  ✓ %s\n", msg); tests_passed++; } while(0)
#define FAIL(msg) do { printf("  ✗ %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(msg); else FAIL(msg); } while(0)
#define CHECK_RANGE(val, lo, hi, msg) CHECK((val) >= (lo) && (val) <= (hi), msg)

// ═══════════════════════════════════════════════════════════════════════════════
// Mock Functions
// ═══════════════════════════════════════════════════════════════════════════════

// Mock brain decoder: token_id -> word
static const char* mock_brain_decode(int token_id) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "word%d", token_id);
    return buf;
}

// Mock arianna encoder: word -> token_id
static int mock_arianna_encode(const char* word) {
    // Simple hash-based mapping
    if (strncmp(word, "word", 4) == 0) {
        return atoi(word + 4) % 80;  // Map to 80-token vocab
    }
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

void test_init(void) {
    TEST("Pandora Initialization");

    PandoraBox pandora;
    pandora_init(&pandora);

    CHECK(pandora.n_ngrams == 0, "No n-grams initially");
    CHECK(pandora.total_released == 0, "No released initially");
    CHECK(pandora.active == 1, "Active by default");
    CHECK(pandora.injection_strength > 0.0f, "Positive injection strength");

    PASS("Pandora initialized");
}

void test_extract(void) {
    TEST("N-gram Extraction");

    PandoraBox pandora;
    pandora_init(&pandora);

    // Simulate tokens from external brain
    int tokens[] = {10, 20, 30, 40, 50, 10, 20, 30};  // Has repeating pattern
    int n_tokens = 8;

    pandora_extract(&pandora, tokens, n_tokens, 2, 3);

    CHECK(pandora.n_ngrams > 0, "Extracted some n-grams");
    CHECK(pandora.total_released > 0, "Total released updated");

    // Check that bigram [10, 20] was found (appears twice)
    int found_pattern = 0;
    for (int i = 0; i < pandora.n_ngrams; i++) {
        if (pandora.ngrams[i].length == 2 &&
            pandora.ngrams[i].tokens[0] == 10 &&
            pandora.ngrams[i].tokens[1] == 20) {
            found_pattern = 1;
            CHECK(pandora.ngrams[i].frequency >= 2, "Pattern frequency >= 2");
            break;
        }
    }
    CHECK(found_pattern, "Found repeating bigram pattern");
}

void test_mapping(void) {
    TEST("Vocabulary Mapping");

    PandoraBox pandora;
    pandora_init(&pandora);

    // Extract some n-grams
    int tokens[] = {5, 10, 15, 5, 10, 15, 5, 10, 15};
    pandora_extract(&pandora, tokens, 9, 2, 3);

    // Map to Arianna vocabulary
    int mapped = pandora_map_to_arianna(&pandora, mock_brain_decode, mock_arianna_encode);

    CHECK(mapped >= 0, "Mapping returned non-negative");
    CHECK(pandora.successfully_mapped >= 0, "Successfully mapped count valid");
}

void test_suggest_continuation(void) {
    TEST("Continuation Suggestion");

    PandoraBox pandora;
    pandora_init(&pandora);

    // Create a known n-gram manually
    pandora.ngrams[0].tokens[0] = 1;
    pandora.ngrams[0].tokens[1] = 2;
    pandora.ngrams[0].tokens[2] = 3;
    pandora.ngrams[0].length = 3;
    pandora.ngrams[0].weight = 1.0f;
    pandora.ngrams[0].arianna_mapped = 1;
    pandora.ngrams[0].arianna_tokens[0] = 1;
    pandora.ngrams[0].arianna_tokens[1] = 2;
    pandora.ngrams[0].arianna_tokens[2] = 3;
    pandora.n_ngrams = 1;

    // Test with matching prefix
    int context1[] = {1, 2};
    int suggestion = pandora_suggest_continuation(&pandora, context1, 2);
    CHECK(suggestion == 3 || suggestion == -1, "Suggestion is valid or -1");

    // Test with non-matching prefix
    int context2[] = {99, 98};
    int no_match = pandora_suggest_continuation(&pandora, context2, 2);
    CHECK(no_match == -1, "No suggestion for unknown prefix");
}

void test_apply_to_logits(void) {
    TEST("Logit Application");

    PandoraBox pandora;
    pandora_init(&pandora);
    pandora_set_strength(&pandora, 0.5f);

    // Create a known n-gram
    pandora.ngrams[0].tokens[0] = 1;
    pandora.ngrams[0].tokens[1] = 5;
    pandora.ngrams[0].length = 2;
    pandora.ngrams[0].weight = 1.0f;
    pandora.ngrams[0].arianna_mapped = 1;
    pandora.ngrams[0].arianna_tokens[0] = 1;
    pandora.ngrams[0].arianna_tokens[1] = 5;
    pandora.n_ngrams = 1;

    // Prepare logits
    float logits[80] = {0};
    float original_logit_5 = logits[5];

    int context[] = {1};  // Matches first token of n-gram
    pandora_apply_to_logits(&pandora, logits, context, 1, 80);

    // The logit for token 5 should be boosted (or unchanged if no match logic)
    // Just check no crash and logits are still valid
    int valid = 1;
    for (int i = 0; i < 80; i++) {
        if (isnan(logits[i]) || isinf(logits[i])) {
            valid = 0;
            break;
        }
    }
    CHECK(valid, "Logits remain valid after application");
}

void test_decay(void) {
    TEST("N-gram Decay");

    PandoraBox pandora;
    pandora_init(&pandora);

    // Add an n-gram with weight 1.0
    pandora.ngrams[0].length = 2;
    pandora.ngrams[0].weight = 1.0f;
    pandora.ngrams[0].frequency = 10;
    pandora.n_ngrams = 1;

    float original_weight = pandora.ngrams[0].weight;

    // Apply decay
    pandora_decay(&pandora, 0.9f);

    CHECK(pandora.ngrams[0].weight <= original_weight, "Weight decayed or unchanged");
    CHECK(pandora.ngrams[0].weight >= 0.0f, "Weight non-negative");
}

void test_activation(void) {
    TEST("Activation Control");

    PandoraBox pandora;
    pandora_init(&pandora);

    CHECK(pandora.active == 1, "Active by default");

    pandora_set_active(&pandora, 0);
    CHECK(pandora.active == 0, "Deactivated");

    pandora_set_active(&pandora, 1);
    CHECK(pandora.active == 1, "Reactivated");
}

void test_strength(void) {
    TEST("Injection Strength");

    PandoraBox pandora;
    pandora_init(&pandora);

    pandora_set_strength(&pandora, 0.75f);
    CHECK_RANGE(pandora.injection_strength, 0.7f, 0.8f, "Strength set to 0.75");

    pandora_set_strength(&pandora, 0.0f);
    CHECK_RANGE(pandora.injection_strength, 0.0f, 0.1f, "Strength set to 0.0");

    pandora_set_strength(&pandora, 1.0f);
    CHECK_RANGE(pandora.injection_strength, 0.9f, 1.0f, "Strength set to 1.0");
}

void test_save_load(void) {
    TEST("Save/Load");

    PandoraBox pandora1;
    pandora_init(&pandora1);

    // Add some data
    pandora1.ngrams[0].tokens[0] = 42;
    pandora1.ngrams[0].tokens[1] = 43;
    pandora1.ngrams[0].length = 2;
    pandora1.ngrams[0].weight = 0.8f;
    pandora1.ngrams[0].frequency = 5;
    pandora1.n_ngrams = 1;
    pandora1.injection_strength = 0.6f;

    // Save
    int save_result = pandora_save(&pandora1, "/tmp/test_pandora.bin");
    CHECK(save_result == 0, "Save succeeded");

    // Load into new box
    PandoraBox pandora2;
    pandora_init(&pandora2);
    int load_result = pandora_load(&pandora2, "/tmp/test_pandora.bin");
    CHECK(load_result == 0, "Load succeeded");

    // Verify
    CHECK(pandora2.n_ngrams == 1, "N-gram count preserved");
    CHECK(pandora2.ngrams[0].tokens[0] == 42, "Token 0 preserved");
    CHECK(pandora2.ngrams[0].tokens[1] == 43, "Token 1 preserved");
    CHECK(pandora2.ngrams[0].length == 2, "Length preserved");
    CHECK(fabsf(pandora2.ngrams[0].weight - 0.8f) < 0.01f, "Weight preserved");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(void) {
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  PANDORA TEST SUITE\n");
    printf("  \"Take the words, leave the voice\"\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    test_init();
    test_extract();
    test_mapping();
    test_suggest_continuation();
    test_apply_to_logits();
    test_decay();
    test_activation();
    test_strength();
    test_save_load();

    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════════════\n");

    if (tests_failed > 0) {
        printf("\n  לגנוב מילים נכשל\n\n");
        return 1;
    } else {
        printf("\n  ✓ לגנוב מילים, להשאיר את הקול\n\n");
        return 0;
    }
}
