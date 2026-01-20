/*
 * test_dialogue_brain.c - Test char-level GPT-2 model
 * Uses external_brain.c with char tokenization instead of BPE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Reuse GPT-2 model structures from external_brain
#include "external_brain.h"

// Exact vocab from nanoGPT arianna training
static const char* CHAR_VOCAB =
    "\n \"'(),-./05:;?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz\xc3\xaf";
// Note: last char is ï (UTF-8: 0xc3 0xaf)

static int char_to_id(char c) {
    // Handle space specially (position 1 in vocab)
    if (c == ' ') return 1;
    // Find char in vocab
    const char* p = strchr(CHAR_VOCAB, c);
    if (p) return (int)(p - CHAR_VOCAB);
    return 1;  // space as default for unknown
}

static char id_to_char(int id) {
    int len = strlen(CHAR_VOCAB);
    if (id >= 0 && id < len) return CHAR_VOCAB[id];
    return '?';
}

// Encode string to token ids
static int char_encode(const char* text, int* out_tokens, int max_len) {
    int n = 0;
    while (*text && n < max_len) {
        out_tokens[n++] = char_to_id(*text);
        text++;
    }
    return n;
}

// Decode token ids to string
static void char_decode(const int* tokens, int n_tokens, char* out, int max_len) {
    int i;
    for (i = 0; i < n_tokens && i < max_len - 1; i++) {
        out[i] = id_to_char(tokens[i]);
    }
    out[i] = '\0';
}

// Sample from logits
static int sample_logits(float* logits, int vocab_size, float temperature) {
    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }

    // Softmax
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

int main(int argc, char** argv) {
    srand(time(NULL));

    const char* weights_path = "weights/dialogue_brain.bin";
    const char* prompt = "What is love?";
    int max_tokens = 200;
    float temperature = 0.8f;

    if (argc > 1) weights_path = argv[1];
    if (argc > 2) prompt = argv[2];
    if (argc > 3) max_tokens = atoi(argv[3]);
    if (argc > 4) temperature = atof(argv[4]);

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("CHAR-LEVEL GPT-2 TEST — Dialogue Brain\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Initialize brain (reusing external_brain structures)
    ExternalBrain brain;
    brain_init(&brain);

    printf("[*] Loading model from: %s\n", weights_path);
    if (brain_load_model(&brain, weights_path) < 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Override tokenizer with char-level
    brain.loaded = 1;  // Skip BPE tokenizer

    printf("[*] Prompt: \"%s\"\n", prompt);
    printf("[*] Temperature: %.2f\n", temperature);
    printf("[*] Max tokens: %d\n\n", max_tokens);

    // Encode prompt
    int tokens[512];
    int n_tokens = char_encode(prompt, tokens, 256);

    printf("───────────────────────────────────────────────────────────────────\n");
    printf("GENERATION:\n");
    printf("───────────────────────────────────────────────────────────────────\n");

    // Print prompt
    printf("%s", prompt);
    fflush(stdout);

    // Reset KV cache
    brain.model.cache_len = 0;

    // Process prompt
    brain_forward(&brain, tokens, n_tokens, 0);
    brain.model.cache_len = n_tokens;

    // Generate
    clock_t start = clock();
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        // Sample
        int next_token = sample_logits(brain.model.logits, brain.model.vocab_size, temperature);

        // Print char
        char c = id_to_char(next_token);
        printf("%c", c);
        fflush(stdout);

        // Add to sequence
        tokens[n_tokens] = next_token;
        n_tokens++;
        generated++;

        // Forward
        brain_forward(&brain, &next_token, 1, n_tokens - 1);
        brain.model.cache_len = n_tokens;

        // Stop on too many newlines
        if (c == '\n') {
            // Count recent newlines
            int nl_count = 0;
            for (int j = n_tokens - 1; j >= 0 && j >= n_tokens - 3; j--) {
                if (id_to_char(tokens[j]) == '\n') nl_count++;
            }
            if (nl_count >= 2) break;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n───────────────────────────────────────────────────────────────────\n");
    printf("\n[*] Generated %d tokens in %.2f seconds (%.1f tok/s)\n",
           generated, elapsed, generated / elapsed);

    brain_free(&brain);
    return 0;
}
