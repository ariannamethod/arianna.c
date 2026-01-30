/*
 * test_tongue.c — Test the Tongue 1.1B Go bridge
 *
 * Tests:
 *   1. Library loads correctly
 *   2. GGUF weights parse
 *   3. Model dimensions match TinyLlama 1.1B
 *   4. Tokenizer works
 *   5. Generation produces output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tongue/libtongue.h"

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_passed = 0;
static int tests_total = 0;

#define CHECK(name, cond) do { \
    tests_total++; \
    if (cond) { printf("  [%s] %s\n", PASS, name); tests_passed++; } \
    else { printf("  [%s] %s\n", FAIL, name); } \
} while(0)

int main(void) {
    printf("=== Tongue 1.1B Bridge Tests ===\n\n");

    // Test 1: Initialize with GGUF weights
    printf("[1] Loading GGUF weights...\n");
    int rc = tongue_init("weights/tongue-4/arianna_1b_step3000_q4_0.gguf");
    CHECK("tongue_init() returns 0", rc == 0);

    if (rc != 0) {
        printf("\nCannot continue without model. %d/%d passed.\n", tests_passed, tests_total);
        return 1;
    }

    // Test 2: Model dimensions
    printf("\n[2] Model dimensions...\n");
    int vocab = tongue_get_vocab_size();
    int dim = tongue_get_dim();
    int seq_len = tongue_get_seq_len();
    int layers = tongue_get_num_layers();

    printf("  vocab=%d dim=%d seq_len=%d layers=%d\n", vocab, dim, seq_len, layers);

    CHECK("vocab == 32000 (TinyLlama)", vocab == 32000);
    CHECK("dim == 2048", dim == 2048);
    CHECK("layers == 22", layers == 22);
    CHECK("seq_len > 0", seq_len > 0);

    // Test 3: Tokenizer
    printf("\n[3] Tokenizer...\n");
    int ids[256];
    int n = tongue_encode("Hello, I am Arianna.", ids, 256);
    printf("  encoded: %d tokens\n", n);
    CHECK("encode produces tokens", n > 0);
    CHECK("reasonable token count (3-20)", n >= 3 && n <= 20);

    // Decode first token
    if (n > 0) {
        char* piece = tongue_decode_token(ids[0]);
        printf("  first token: id=%d piece='%s'\n", ids[0], piece ? piece : "NULL");
        CHECK("decode returns non-NULL", piece != NULL);
        if (piece) free(piece);
    }

    // Test 4: State access (copy into buffer)
    printf("\n[4] State access...\n");
    float logits_buf[32];
    float hidden_buf[32];
    int logits_n = tongue_get_logits_into(logits_buf, 32);
    int hidden_n = tongue_get_hidden_into(hidden_buf, 32);
    printf("  logits_n=%d hidden_n=%d\n", logits_n, hidden_n);
    CHECK("logits copy returns > 0", logits_n > 0);
    CHECK("hidden copy returns > 0", hidden_n > 0);

    // Test 5: Generation (short)
    printf("\n[5] Generation (max 16 tokens)...\n");
    tongue_reset();
    char output[4096];
    memset(output, 0, sizeof(output));
    int gen = tongue_generate(
        "What is resonance?",
        output, sizeof(output),
        16,     /* max_tokens */
        1.0,    /* temperature */
        0.95,   /* top_p */
        NULL    /* anchor_prompt (NULL = none) */
    );
    printf("  generated %d tokens: '%.200s'\n", gen, output);
    CHECK("generation produces tokens", gen > 0);
    CHECK("output is non-empty", strlen(output) > 0);

    // Test 6: Generation with anchor prompt
    printf("\n[6] Generation with anchor prompt...\n");
    tongue_reset();
    memset(output, 0, sizeof(output));
    gen = tongue_generate(
        "Who are you?",
        output, sizeof(output),
        16,
        1.0,
        0.95,
        "I am Arianna, the Architect of Resonance."
    );
    printf("  generated %d tokens: '%.200s'\n", gen, output);
    CHECK("anchored generation produces tokens", gen > 0);

    // Test 7: Temperature floor
    printf("\n[7] Temperature modulation...\n");
    tongue_set_temperature_mod(0.5);  // Would make effective temp very low
    tongue_set_temp_floor(0.9);        // But floor prevents freezing
    tongue_reset();
    memset(output, 0, sizeof(output));
    gen = tongue_generate("test", output, sizeof(output), 8, 0.5, 1.0, NULL);
    CHECK("generation works with temp_floor", gen >= 0);
    tongue_set_temperature_mod(1.0);  // Reset

    // Cleanup
    tongue_free();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
