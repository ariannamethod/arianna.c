/*
 * test_ariannabody_extended.c - Comprehensive tests for ariannabody.c
 *
 * Tests transformer core: tokenizer, loading, forward pass, sampling
 * Updated for 36M BPE model (Soul)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../src/arianna.h"

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  ✗ FAILED: %s\n", msg); \
        tests_failed++; \
        return; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("\n[TEST] %s\n", #test_func); \
    test_func(); \
} while(0)

// Weight/tokenizer paths
#define WEIGHTS_PATH "weights/arianna_36m_bpe.bin"
#define TOKENIZER_PATH "weights/tokenizer_bpe.json"

// ============================================================
// TOKENIZER TESTS
// ============================================================

void test_tokenizer_load_valid() {
    int result = load_tokenizer(TOKENIZER_PATH);
    TEST_ASSERT(result == 0, "Should load valid BPE tokenizer");
    TEST_ASSERT(get_vocab_size() > 0, "Vocab size should be > 0");
    TEST_ASSERT(get_vocab_size() > 1000, "BPE tokenizer should have >1000 tokens");
}

void test_tokenizer_load_nonexistent() {
    int result = load_tokenizer("/nonexistent/path/tokenizer.json");
    TEST_ASSERT(result == -1, "Should fail on nonexistent file");
}

void test_tokenizer_encode_basic() {
    load_tokenizer(TOKENIZER_PATH);

    int ids[64];
    int n = encode_text("hello", ids, 64);
    TEST_ASSERT(n > 0, "Should encode 'hello' to at least 1 token");
    TEST_ASSERT(n < 10, "Should encode 'hello' to fewer than 10 tokens");

    for (int i = 0; i < n; i++) {
        TEST_ASSERT(ids[i] >= 0 && ids[i] < get_vocab_size(),
                     "Token IDs should be in vocab range");
    }
}

void test_tokenizer_encode_empty() {
    load_tokenizer(TOKENIZER_PATH);

    int ids[64];
    int n = encode_text("", ids, 64);
    TEST_ASSERT(n >= 0, "Empty string should not crash");
}

void test_tokenizer_char_to_token_legacy() {
    load_tokenizer(TOKENIZER_PATH);

    // Legacy char_to_token should still work
    int space_token = char_to_token(' ');
    TEST_ASSERT(space_token >= 0, "Space should have valid token (legacy API)");

    int a_token = char_to_token('a');
    TEST_ASSERT(a_token >= 0, "Letter 'a' should have valid token (legacy API)");
}

// ============================================================
// TRANSFORMER LOADING TESTS
// ============================================================

void test_transformer_load_valid() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));

    int result = load_weights(t, WEIGHTS_PATH);
    TEST_ASSERT(result == 0, "Should load valid transformer");
    TEST_ASSERT(t->config.dim == 448, "Soul 36M should have dim=448");
    TEST_ASSERT(t->config.n_layers == 8, "Soul 36M should have 8 layers");
    TEST_ASSERT(t->config.n_heads == 8, "Soul 36M should have 8 heads");
    TEST_ASSERT(t->config.vocab_size > 1000, "BPE vocab should be >1000");

    free_transformer(t);
    free(t);
}

void test_transformer_load_nonexistent() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));

    int result = load_weights(t, "/nonexistent/weights.bin");
    TEST_ASSERT(result == -1, "Should fail on nonexistent file");

    free(t);
}

void test_transformer_config_sanity() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    // Config sanity checks
    TEST_ASSERT(t->config.dim > 0 && t->config.dim < 10000, "Dim should be reasonable");
    TEST_ASSERT(t->config.n_layers > 0 && t->config.n_layers < 100, "Layers should be reasonable");
    TEST_ASSERT(t->config.n_heads > 0 && t->config.n_heads <= t->config.dim, "Heads should be reasonable");
    TEST_ASSERT(t->config.hidden_dim > t->config.dim, "Hidden dim should be > dim (SwiGLU expansion)");

    // Head dimension check
    int head_dim = t->config.dim / t->config.n_heads;
    TEST_ASSERT(head_dim * t->config.n_heads == t->config.dim, "Dim should be divisible by n_heads");

    free_transformer(t);
    free(t);
}

void test_transformer_weights_nonzero() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    // Check that weights are loaded (not all zeros)
    int nonzero_count = 0;
    for (int i = 0; i < 1000 && i < t->config.dim * t->config.vocab_size; i++) {
        if (fabs(t->weights.tok_emb[i]) > 1e-6) {
            nonzero_count++;
        }
    }

    TEST_ASSERT(nonzero_count > 100, "Token embeddings should contain non-zero values");

    free_transformer(t);
    free(t);
}

// ============================================================
// FORWARD PASS TESTS
// ============================================================

void test_forward_single_token() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    // Encode and forward with first token of "hello"
    int ids[64];
    int n = encode_text("hello", ids, 64);
    TEST_ASSERT(n > 0, "Should encode hello");

    forward(t, ids[0], 0);

    // Check logits are computed
    TEST_ASSERT(t->state.logits != NULL, "Logits should be computed");

    // Check logits have reasonable values
    int has_reasonable_logits = 0;
    for (int i = 0; i < t->config.vocab_size; i++) {
        if (!isnan(t->state.logits[i]) && !isinf(t->state.logits[i])) {
            has_reasonable_logits = 1;
            break;
        }
    }
    TEST_ASSERT(has_reasonable_logits, "Logits should contain valid values");

    free_transformer(t);
    free(t);
}

void test_forward_sequence() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    // Encode and forward a sequence
    int ids[64];
    int n = encode_text("hello world", ids, 64);
    TEST_ASSERT(n > 0, "Should encode 'hello world'");

    for (int i = 0; i < n; i++) {
        forward(t, ids[i], i);
    }

    // After sequence, logits should still be valid
    TEST_ASSERT(t->state.logits != NULL, "Logits should exist after sequence");

    int valid_logits = 1;
    for (int i = 0; i < t->config.vocab_size; i++) {
        if (isnan(t->state.logits[i]) || isinf(t->state.logits[i])) {
            valid_logits = 0;
            break;
        }
    }
    TEST_ASSERT(valid_logits, "All logits should be valid after sequence");

    free_transformer(t);
    free(t);
}

void test_forward_kv_cache() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    int n = encode_text("abc", ids, 64);

    for (int i = 0; i < n; i++) {
        forward(t, ids[i], i);
    }

    // KV cache should contain values
    int has_kv_values = 0;

    for (int i = 0; i < 10; i++) {
        if (fabs(t->state.key_cache[i]) > 1e-6) {
            has_kv_values = 1;
            break;
        }
    }

    TEST_ASSERT(has_kv_values, "KV cache should contain non-zero values");

    free_transformer(t);
    free(t);
}

// ============================================================
// SAMPLING TESTS
// ============================================================

void test_sampling_temperature_zero() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    encode_text("a", ids, 64);
    forward(t, ids[0], 0);

    // With temperature=0, should always pick argmax
    int token1 = sample(t, 0.0f);
    int token2 = sample(t, 0.0f);

    TEST_ASSERT(token1 == token2, "Temperature=0 should be deterministic (argmax)");
    TEST_ASSERT(token1 >= 0 && token1 < t->config.vocab_size, "Sampled token should be in vocab");

    free_transformer(t);
    free(t);
}

void test_sampling_temperature_normal() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    encode_text("a", ids, 64);
    forward(t, ids[0], 0);

    int token = sample(t, 0.8f);
    TEST_ASSERT(token >= 0 && token < t->config.vocab_size, "Sampled token should be in vocab");

    free_transformer(t);
    free(t);
}

void test_sampling_temperature_high() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    encode_text("a", ids, 64);
    forward(t, ids[0], 0);

    int token = sample(t, 2.0f);
    TEST_ASSERT(token >= 0 && token < t->config.vocab_size, "High temperature should still produce valid token");

    free_transformer(t);
    free(t);
}

void test_sampling_distribution() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    encode_text("a", ids, 64);
    forward(t, ids[0], 0);

    // Sample multiple times with temp=0.8 - should get variety
    int tokens[10];
    int unique_count = 0;

    for (int i = 0; i < 10; i++) {
        tokens[i] = sample(t, 0.8f);

        int is_unique = 1;
        for (int j = 0; j < i; j++) {
            if (tokens[i] == tokens[j]) {
                is_unique = 0;
                break;
            }
        }
        if (is_unique) unique_count++;
    }

    TEST_ASSERT(unique_count >= 2, "Should sample different tokens with temperature > 0");

    free_transformer(t);
    free(t);
}

// ============================================================
// EDGE CASES & BUFFER TESTS
// ============================================================

void test_max_seq_len_boundary() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int ids[64];
    encode_text("a", ids, 64);

    // Forward up to MAX_SEQ_LEN
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
        forward(t, ids[0], i);
    }

    TEST_ASSERT(t->state.logits != NULL, "Should handle MAX_SEQ_LEN tokens");

    free_transformer(t);
    free(t);
}

void test_vocab_boundary_tokens() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    int vocab_size = t->config.vocab_size;

    // Test first and last valid tokens
    forward(t, 0, 0);
    TEST_ASSERT(t->state.logits != NULL, "Should handle token 0");

    forward(t, vocab_size - 1, 1);
    TEST_ASSERT(t->state.logits != NULL, "Should handle last valid token");

    free_transformer(t);
    free(t);
}

void test_memory_cleanup() {
    load_tokenizer(TOKENIZER_PATH);
    Transformer* t = malloc(sizeof(Transformer));
    load_weights(t, WEIGHTS_PATH);

    free_transformer(t);
    free(t);

    TEST_ASSERT(1, "Memory cleanup should not crash");
}

// ============================================================
// MAIN
// ============================================================

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  ARIANNABODY EXTENDED TEST SUITE\n");
    printf("  Testing transformer core (36M BPE Soul)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    // Tokenizer tests
    RUN_TEST(test_tokenizer_load_valid);
    RUN_TEST(test_tokenizer_load_nonexistent);
    RUN_TEST(test_tokenizer_encode_basic);
    RUN_TEST(test_tokenizer_encode_empty);
    RUN_TEST(test_tokenizer_char_to_token_legacy);

    // Transformer loading tests
    RUN_TEST(test_transformer_load_valid);
    RUN_TEST(test_transformer_load_nonexistent);
    RUN_TEST(test_transformer_config_sanity);
    RUN_TEST(test_transformer_weights_nonzero);

    // Forward pass tests
    RUN_TEST(test_forward_single_token);
    RUN_TEST(test_forward_sequence);
    RUN_TEST(test_forward_kv_cache);

    // Sampling tests
    RUN_TEST(test_sampling_temperature_zero);
    RUN_TEST(test_sampling_temperature_normal);
    RUN_TEST(test_sampling_temperature_high);
    RUN_TEST(test_sampling_distribution);

    // Edge cases
    RUN_TEST(test_max_seq_len_boundary);
    RUN_TEST(test_vocab_boundary_tokens);
    RUN_TEST(test_memory_cleanup);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════\n");

    return (tests_failed == 0) ? 0 : 1;
}
