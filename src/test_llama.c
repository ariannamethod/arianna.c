/*
 * test_llama.c - Simple test for Llama 3.5 Arianna Edition
 */

#include "arianna.h"
#include <time.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <weights.bin> <tokenizer.json> [prompt]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    // Load tokenizer
    if (load_tokenizer(argv[2]) != 0) {
        printf("Error loading tokenizer\n");
        return 1;
    }

    // Load model
    Transformer t;
    memset(&t, 0, sizeof(Transformer));

    if (load_weights(&t, argv[1]) != 0) {
        printf("Error loading weights\n");
        return 1;
    }

    // Get prompt
    const char* prompt = argc > 3 ? argv[3] : "Q: What is consciousness?\nA:";

    printf("\n=== Llama 3.5 Arianna Edition Test ===\n");
    printf("Prompt: %s\n", prompt);
    printf("==========================================\n");

    clock_t start = clock();
    generate(&t, prompt, 150, 0.8f);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n==========================================\n");
    printf("Generated in %.2f seconds\n", elapsed);

    free_transformer(&t);
    return 0;
}
