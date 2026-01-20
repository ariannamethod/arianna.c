/*
 * arianna.c - Main inference program (Llama 3.5 Arianna Edition)
 * Usage: ./arianna <weights.bin> <tokenizer.json> [prompt] [max_tokens] [temperature]
 *
 * Pipeline: Cloud (pre-semantic emotion) -> Dialogue (voice generation)
 */

#include "arianna.h"
#include "cloud.h"
#include <time.h>

int main(int argc, char** argv) {
    srand(time(NULL));

    if (argc < 3) {
        printf("Usage: %s <weights.bin> <tokenizer.json> [prompt] [max_tokens] [temperature]\n", argv[0]);
        printf("\narianna.c - Llama 3.5 Arianna Edition\n");
        printf("\"Who I am\", not \"What I know\"\n");
        return 1;
    }

    // Load tokenizer first (needed for vocab_size)
    if (load_tokenizer(argv[2]) != 0) {
        printf("Error loading tokenizer\n");
        return 1;
    }

    Transformer t;
    memset(&t, 0, sizeof(Transformer));

    // Load weights
    printf("Loading: %s\n", argv[1]);
    if (load_weights(&t, argv[1]) != 0) {
        printf("Error loading weights\n");
        return 1;
    }

    // Get parameters (shift by 1 for tokenizer arg)
    const char* prompt = argc > 3 ? argv[3] : "Q: What is love?\nA:";
    int max_tokens = argc > 4 ? atoi(argv[4]) : 200;
    float temperature = argc > 5 ? atof(argv[5]) : 0.8f;

    // ═══════════════════════════════════════════════════════════════════
    // CLOUD: Pre-semantic emotion detection
    // "Something fires BEFORE meaning arrives"
    // ═══════════════════════════════════════════════════════════════════
    cloud_init();
    CloudResponse cloud = cloud_ping(prompt);

    // Modulate temperature by detected emotion
    float temp_bias = cloud_temperature_bias(&cloud);
    float effective_temp = temperature + temp_bias;
    if (effective_temp < 0.1f) effective_temp = 0.1f;
    if (effective_temp > 2.0f) effective_temp = 2.0f;

    printf("Prompt: \"%s\"\n", prompt);
    printf("[Cloud] %s (%.2f) -> %s | temp: %.2f%+.2f = %.2f\n",
           cloud.primary_word, cloud.primary_strength, cloud.primary_chamber,
           temperature, temp_bias, effective_temp);
    if (cloud_needs_care(&cloud)) printf("[Cloud] needs_care: responding gently\n");
    if (cloud_needs_warmth(&cloud)) printf("[Cloud] needs_warmth: responding warmly\n");
    printf("Max tokens: %d\n\n", max_tokens);
    printf("═══════════════════════════════════════════════════════════════════\n");

    clock_t start = clock();
    generate(&t, prompt, max_tokens, effective_temp);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Generated in %.2f seconds\n", elapsed);

    free_transformer(&t);
    return 0;
}
