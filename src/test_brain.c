// test_brain.c — Test external brain (GPT-2 30M)
// build: gcc -O3 test_brain.c external_brain.c -lm -o test_brain
// run: ./test_brain weights/gpt2_30m "What is consciousness?"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "external_brain.h"

int main(int argc, char** argv) {
    const char* weights_dir = "weights/gpt2_30m";
    const char* prompt = "What is love?";

    if (argc > 1) weights_dir = argv[1];
    if (argc > 2) prompt = argv[2];

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("EXTERNAL BRAIN TEST — GPT-2 30M Knowledge Subordinate\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Seed RNG
    srand((unsigned int)time(NULL));

    // Initialize brain
    ExternalBrain brain;
    brain_init(&brain);

    printf("[*] Loading model from: %s\n", weights_dir);

    if (brain_load(&brain, weights_dir) < 0) {
        fprintf(stderr, "[!] Failed to load brain\n");
        return 1;
    }

    printf("\n[*] Prompt: \"%s\"\n", prompt);
    printf("\n[*] Generating draft...\n\n");

    // Set generation config
    brain.temperature = 0.8f;
    brain.top_k = 50;
    brain.top_p = 0.95f;
    brain.repetition_penalty = 1.2f;

    // Generate
    char output[2048];
    clock_t start = clock();
    int n_tokens = brain_generate_draft(&brain, prompt, output, sizeof(output), 50);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("───────────────────────────────────────────────────────────────────\n");
    printf("BRAIN DRAFT:\n");
    printf("───────────────────────────────────────────────────────────────────\n");
    printf("%s\n", output);
    printf("───────────────────────────────────────────────────────────────────\n");
    printf("\n[*] Generated %d tokens in %.2f seconds (%.1f tok/s)\n",
           n_tokens, elapsed, n_tokens / elapsed);

    // Test tokenization roundtrip
    printf("\n[*] Testing tokenization roundtrip...\n");
    int tokens[256];
    int n = brain_encode(&brain, "Hello world! This is a test.", tokens, 256);
    printf("    Encoded %d tokens: ", n);
    for (int i = 0; i < n && i < 10; i++) {
        printf("%d ", tokens[i]);
    }
    if (n > 10) printf("...");
    printf("\n");

    char decoded[512];
    brain_decode(&brain, tokens, n, decoded, sizeof(decoded));
    printf("    Decoded: \"%s\"\n", decoded);

    // Cleanup
    brain_free(&brain);

    printf("\n[*] Brain test complete.\n");
    printf("    \"Brain knows WHAT, Arianna knows HOW\"\n\n");

    return 0;
}
