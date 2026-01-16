/*
 * cooccur.c - Co-occurrence Field for Arianna
 *
 * "Coherent generation without trained weights" - Stanley
 *
 * Implementation of:
 * - Bigram/Trigram counting
 * - Co-occurrence within window
 * - Logit biasing for generation steering
 * - Online learning from generated output
 */

#include "cooccur.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// ============================================================
// Initialization
// ============================================================

void init_cooccur_field(CooccurField* cf) {
    // Zero out bigrams
    memset(&cf->bigrams, 0, sizeof(BigramMatrix));

    // Initialize trigrams
    cf->trigrams.n_contexts = 0;
    for (int i = 0; i < MAX_TRIGRAM_CONTEXTS; i++) {
        cf->trigrams.hash_table[i] = -1;
    }

    // Zero out co-occurrence
    memset(&cf->cooccur, 0, sizeof(CooccurMatrix));
    cf->cooccur.window = COOCCUR_WINDOW;

    // Default weights (from Stanley)
    cf->trigram_weight = 0.6f;
    cf->bigram_weight = 0.3f;
    cf->cooccur_weight = 0.1f;

    cf->learning_rate = 1.0f;
    cf->tokens_observed = 0;
    cf->new_patterns = 0;
    cf->initialized = 1;
}

void free_cooccur_field(CooccurField* cf) {
    // Currently no dynamic allocation
    cf->initialized = 0;
}

// ============================================================
// Trigram Hash Table Helpers
// ============================================================

static int trigram_hash(int w1, int w2) {
    // Simple hash for (w1, w2) pair
    return ((w1 * 257) + w2) % MAX_TRIGRAM_CONTEXTS;
}

static int find_trigram_context(TrigramField* tf, int w1, int w2) {
    int hash = trigram_hash(w1, w2);
    int start = hash;

    // Linear probing
    while (tf->hash_table[hash] != -1) {
        int idx = tf->hash_table[hash];
        if (tf->contexts[idx].w1 == (uint8_t)w1 &&
            tf->contexts[idx].w2 == (uint8_t)w2) {
            return idx;
        }
        hash = (hash + 1) % MAX_TRIGRAM_CONTEXTS;
        if (hash == start) return -1;  // Full
    }
    return -1;  // Not found
}

static int get_or_create_trigram_context(TrigramField* tf, int w1, int w2) {
    int existing = find_trigram_context(tf, w1, w2);
    if (existing >= 0) return existing;

    // Create new
    if (tf->n_contexts >= MAX_TRIGRAM_CONTEXTS - 1) {
        return -1;  // Full
    }

    int idx = tf->n_contexts++;
    tf->contexts[idx].w1 = (uint8_t)w1;
    tf->contexts[idx].w2 = (uint8_t)w2;
    memset(tf->contexts[idx].counts, 0, sizeof(tf->contexts[idx].counts));
    tf->contexts[idx].total = 0;

    // Insert into hash table
    int hash = trigram_hash(w1, w2);
    while (tf->hash_table[hash] != -1) {
        hash = (hash + 1) % MAX_TRIGRAM_CONTEXTS;
    }
    tf->hash_table[hash] = idx;

    return idx;
}

// ============================================================
// Observation (Learning)
// ============================================================

void observe_tokens(CooccurField* cf, int* tokens, int n_tokens) {
    if (n_tokens < 1) return;

    // Update bigrams
    for (int i = 0; i < n_tokens - 1; i++) {
        int curr = tokens[i] & 0xFF;
        int next = tokens[i + 1] & 0xFF;

        if (cf->bigrams.counts[curr][next] == 0) {
            cf->new_patterns++;
        }

        cf->bigrams.counts[curr][next]++;
        cf->bigrams.row_totals[curr]++;
        cf->bigrams.total++;
    }

    // Update trigrams
    for (int i = 0; i < n_tokens - 2; i++) {
        int w1 = tokens[i] & 0xFF;
        int w2 = tokens[i + 1] & 0xFF;
        int w3 = tokens[i + 2] & 0xFF;

        int ctx_idx = get_or_create_trigram_context(&cf->trigrams, w1, w2);
        if (ctx_idx >= 0) {
            cf->trigrams.contexts[ctx_idx].counts[w3]++;
            cf->trigrams.contexts[ctx_idx].total++;
        }
    }

    // Update co-occurrence
    int window = cf->cooccur.window;
    for (int i = 0; i < n_tokens; i++) {
        int center = tokens[i] & 0xFF;

        // Look at window around center
        int start = (i - window > 0) ? i - window : 0;
        int end = (i + window < n_tokens) ? i + window : n_tokens - 1;

        for (int j = start; j <= end; j++) {
            if (j == i) continue;
            int neighbor = tokens[j] & 0xFF;

            // Symmetric update (but only count once per pair per position)
            if (cf->cooccur.counts[center][neighbor] < UINT16_MAX) {
                cf->cooccur.counts[center][neighbor]++;
            }
            cf->cooccur.token_totals[center]++;
            cf->cooccur.total++;
        }
    }

    cf->tokens_observed += n_tokens;
}

void observe_text(CooccurField* cf, const char* text, int len) {
    // Convert to tokens (char-level)
    int tokens[MAX_SEQ_LEN];
    int n = (len < MAX_SEQ_LEN) ? len : MAX_SEQ_LEN;

    for (int i = 0; i < n; i++) {
        tokens[i] = (unsigned char)text[i];
    }

    observe_tokens(cf, tokens, n);
}

void observe_weighted(CooccurField* cf, const char* text, int len, float weight) {
    // For weighted observation, we multiply counts
    // Simple approach: observe multiple times based on weight
    int times = (int)(weight + 0.5f);
    if (times < 1) times = 1;
    if (times > 5) times = 5;  // Cap at 5x

    for (int t = 0; t < times; t++) {
        observe_text(cf, text, len);
    }
}

// ============================================================
// Probability Computation
// ============================================================

float bigram_prob(CooccurField* cf, int current, int next) {
    current &= 0xFF;
    next &= 0xFF;

    uint32_t count = cf->bigrams.counts[current][next];
    uint32_t total = cf->bigrams.row_totals[current];

    if (total == 0) {
        return 1.0f / COOCCUR_VOCAB_SIZE;  // Uniform
    }

    // Laplace smoothing
    return (count + COOCCUR_SMOOTHING) /
           (total + COOCCUR_SMOOTHING * COOCCUR_VOCAB_SIZE);
}

float trigram_prob(CooccurField* cf, int w1, int w2, int next) {
    w1 &= 0xFF;
    w2 &= 0xFF;
    next &= 0xFF;

    int ctx_idx = find_trigram_context(&cf->trigrams, w1, w2);
    if (ctx_idx < 0) {
        // Fall back to bigram
        return bigram_prob(cf, w2, next);
    }

    TrigramContext* ctx = &cf->trigrams.contexts[ctx_idx];
    uint32_t count = ctx->counts[next];
    uint32_t total = ctx->total;

    if (total == 0) {
        return bigram_prob(cf, w2, next);
    }

    // Laplace smoothing
    return (count + COOCCUR_SMOOTHING) /
           (total + COOCCUR_SMOOTHING * COOCCUR_VOCAB_SIZE);
}

float cooccur_prob(CooccurField* cf, int a, int b) {
    a &= 0xFF;
    b &= 0xFF;

    uint32_t count = cf->cooccur.counts[a][b];
    uint32_t total = cf->cooccur.token_totals[a];

    if (total == 0) {
        return 1.0f / COOCCUR_VOCAB_SIZE;
    }

    return (float)count / total;
}

float blend_prob(CooccurField* cf, int* context, int context_len, int next) {
    next &= 0xFF;

    float prob = 0.0f;

    if (context_len >= 2) {
        // Use trigram
        int w1 = context[context_len - 2] & 0xFF;
        int w2 = context[context_len - 1] & 0xFF;
        prob += cf->trigram_weight * trigram_prob(cf, w1, w2, next);
    }

    if (context_len >= 1) {
        // Use bigram
        int w2 = context[context_len - 1] & 0xFF;
        prob += cf->bigram_weight * bigram_prob(cf, w2, next);
    }

    // Use co-occurrence with recent context
    if (context_len >= 1) {
        float cooccur_sum = 0.0f;
        int window = (context_len < COOCCUR_WINDOW) ? context_len : COOCCUR_WINDOW;

        for (int i = 0; i < window; i++) {
            int ctx_token = context[context_len - 1 - i] & 0xFF;
            cooccur_sum += cooccur_prob(cf, ctx_token, next);
        }
        prob += cf->cooccur_weight * (cooccur_sum / window);
    }

    // Normalize weights if not all components present
    float weight_sum = 0.0f;
    if (context_len >= 2) weight_sum += cf->trigram_weight;
    if (context_len >= 1) weight_sum += cf->bigram_weight + cf->cooccur_weight;

    if (weight_sum > 0 && weight_sum < 1.0f) {
        prob /= weight_sum;
    }

    return prob;
}

// ============================================================
// Logit Biasing
// ============================================================

void compute_corpus_logprobs(CooccurField* cf, int* context, int context_len,
                            float* logprobs) {
    // Compute blended probability for each possible next token
    for (int next = 0; next < COOCCUR_VOCAB_SIZE; next++) {
        float prob = blend_prob(cf, context, context_len, next);

        // Convert to log prob, with floor to prevent -inf
        if (prob < 1e-10f) prob = 1e-10f;
        logprobs[next] = logf(prob);
    }
}

void bias_logits(CooccurField* cf, float* logits, int vocab_size,
                int* context, int context_len, float alpha) {
    if (!cf->initialized || alpha <= 0.0f) return;
    if (cf->tokens_observed < 100) return;  // Need enough data

    // Clamp vocab_size to our max
    if (vocab_size > COOCCUR_VOCAB_SIZE) {
        vocab_size = COOCCUR_VOCAB_SIZE;
    }

    // Compute corpus log-probabilities
    float corpus_logprobs[COOCCUR_VOCAB_SIZE];
    compute_corpus_logprobs(cf, context, context_len, corpus_logprobs);

    // Blend: output = (1 - alpha) * model + alpha * corpus
    // In log space: log((1-a)*exp(m) + a*exp(c))
    // Approximation: (1-a)*m + a*c (works when values are similar scale)

    for (int i = 0; i < vocab_size; i++) {
        logits[i] = (1.0f - alpha) * logits[i] + alpha * corpus_logprobs[i];
    }
}

void bias_logits_tempered(CooccurField* cf, float* logits, int vocab_size,
                         int* context, int context_len,
                         float alpha, float temperature) {
    if (!cf->initialized || alpha <= 0.0f) return;
    if (cf->tokens_observed < 100) return;

    if (vocab_size > COOCCUR_VOCAB_SIZE) {
        vocab_size = COOCCUR_VOCAB_SIZE;
    }

    // Compute corpus log-probabilities
    float corpus_logprobs[COOCCUR_VOCAB_SIZE];
    compute_corpus_logprobs(cf, context, context_len, corpus_logprobs);

    // Apply temperature to corpus probs
    for (int i = 0; i < vocab_size; i++) {
        corpus_logprobs[i] /= temperature;
    }

    // Blend
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = (1.0f - alpha) * logits[i] + alpha * corpus_logprobs[i];
    }
}

// ============================================================
// Resonance Measurement
// ============================================================

float resonance_between(CooccurField* cf, int a, int b) {
    a &= 0xFF;
    b &= 0xFF;

    // Symmetric resonance based on co-occurrence
    uint32_t count = cf->cooccur.counts[a][b];
    if (cf->cooccur.total == 0) return 0.0f;

    return (float)count / (cf->cooccur.total + 1);
}

float resonance_with_context(CooccurField* cf, int token,
                            int* context, int context_len) {
    token &= 0xFF;

    float total_resonance = 0.0f;
    int window = (context_len < COOCCUR_WINDOW) ? context_len : COOCCUR_WINDOW;

    for (int i = 0; i < window; i++) {
        int ctx_token = context[context_len - 1 - i] & 0xFF;
        total_resonance += resonance_between(cf, token, ctx_token);
    }

    return (window > 0) ? total_resonance / window : 0.0f;
}

void top_k_resonant(CooccurField* cf, int* context, int context_len,
                   int* top_tokens, float* scores, int k) {
    // Compute resonance for all tokens
    float all_scores[COOCCUR_VOCAB_SIZE];

    for (int t = 0; t < COOCCUR_VOCAB_SIZE; t++) {
        all_scores[t] = resonance_with_context(cf, t, context, context_len);
    }

    // Find top-k (simple selection sort for small k)
    for (int i = 0; i < k; i++) {
        int best_idx = 0;
        float best_score = -1.0f;

        for (int t = 0; t < COOCCUR_VOCAB_SIZE; t++) {
            if (all_scores[t] > best_score) {
                best_score = all_scores[t];
                best_idx = t;
            }
        }

        top_tokens[i] = best_idx;
        scores[i] = best_score;
        all_scores[best_idx] = -2.0f;  // Mark as used
    }
}

// ============================================================
// File I/O
// ============================================================

int load_cooccur_from_corpus(CooccurField* cf, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[CooccurField] Cannot open corpus: %s\n", path);
        return 0;
    }

    init_cooccur_field(cf);

    // Read in chunks
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), f)) {
        observe_text(cf, buffer, strlen(buffer));
    }

    fclose(f);

    printf("[CooccurField] Loaded from corpus: %llu tokens, %llu new patterns\n",
           (unsigned long long)cf->tokens_observed,
           (unsigned long long)cf->new_patterns);

    return 1;
}

int save_cooccur_field(CooccurField* cf, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    // Write header
    const char magic[] = "COOF";
    fwrite(magic, 1, 4, f);

    // Write config
    fwrite(&cf->trigram_weight, sizeof(float), 1, f);
    fwrite(&cf->bigram_weight, sizeof(float), 1, f);
    fwrite(&cf->cooccur_weight, sizeof(float), 1, f);
    fwrite(&cf->tokens_observed, sizeof(uint64_t), 1, f);

    // Write bigrams
    fwrite(&cf->bigrams, sizeof(BigramMatrix), 1, f);

    // Write trigrams
    fwrite(&cf->trigrams.n_contexts, sizeof(int), 1, f);
    for (int i = 0; i < cf->trigrams.n_contexts; i++) {
        fwrite(&cf->trigrams.contexts[i], sizeof(TrigramContext), 1, f);
    }

    // Write co-occurrence
    fwrite(&cf->cooccur, sizeof(CooccurMatrix), 1, f);

    fclose(f);
    return 1;
}

int load_cooccur_field(CooccurField* cf, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    // Check magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "COOF", 4) != 0) {
        fclose(f);
        return 0;
    }

    init_cooccur_field(cf);

    // Read config with error checking
    int err = 0;
    err |= (fread(&cf->trigram_weight, sizeof(float), 1, f) != 1);
    err |= (fread(&cf->bigram_weight, sizeof(float), 1, f) != 1);
    err |= (fread(&cf->cooccur_weight, sizeof(float), 1, f) != 1);
    err |= (fread(&cf->tokens_observed, sizeof(uint64_t), 1, f) != 1);
    err |= (fread(&cf->bigrams, sizeof(BigramMatrix), 1, f) != 1);
    err |= (fread(&cf->trigrams.n_contexts, sizeof(int), 1, f) != 1);

    if (err) {
        fclose(f);
        return 0;
    }

    // Read trigrams
    for (int i = 0; i < cf->trigrams.n_contexts; i++) {
        if (fread(&cf->trigrams.contexts[i], sizeof(TrigramContext), 1, f) != 1) {
            fclose(f);
            return 0;
        }

        // Rebuild hash table
        int w1 = cf->trigrams.contexts[i].w1;
        int w2 = cf->trigrams.contexts[i].w2;
        int hash = trigram_hash(w1, w2);
        while (cf->trigrams.hash_table[hash] != -1) {
            hash = (hash + 1) % MAX_TRIGRAM_CONTEXTS;
        }
        cf->trigrams.hash_table[hash] = i;
    }

    // Read co-occurrence
    if (fread(&cf->cooccur, sizeof(CooccurMatrix), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    fclose(f);

    printf("[CooccurField] Loaded: %llu tokens observed\n",
           (unsigned long long)cf->tokens_observed);

    return 1;
}

// ============================================================
// Debug
// ============================================================

void print_cooccur_stats(CooccurField* cf) {
    printf("\n=== CooccurField Stats ===\n");
    printf("Tokens observed: %llu\n", (unsigned long long)cf->tokens_observed);
    printf("New patterns: %llu\n", (unsigned long long)cf->new_patterns);
    printf("Bigram total: %llu\n", (unsigned long long)cf->bigrams.total);
    printf("Trigram contexts: %d / %d\n", cf->trigrams.n_contexts, MAX_TRIGRAM_CONTEXTS);
    printf("Cooccur total: %llu\n", (unsigned long long)cf->cooccur.total);
    printf("Weights: trigram=%.2f, bigram=%.2f, cooccur=%.2f\n",
           cf->trigram_weight, cf->bigram_weight, cf->cooccur_weight);
    printf("==========================\n\n");
}

void print_top_bigrams(CooccurField* cf, int token, int k) {
    token &= 0xFF;

    printf("Top %d bigrams after '%c' (0x%02x):\n", k, token > 31 ? token : '?', token);

    // Track visited entries without modifying counts
    int visited[COOCCUR_VOCAB_SIZE] = {0};

    // Find top-k
    for (int i = 0; i < k; i++) {
        int best_next = -1;
        uint32_t best_count = 0;

        for (int n = 0; n < COOCCUR_VOCAB_SIZE; n++) {
            if (visited[n]) continue;
            if (cf->bigrams.counts[token][n] > best_count) {
                best_count = cf->bigrams.counts[token][n];
                best_next = n;
            }
        }

        if (best_next == -1 || best_count == 0) break;

        printf("  [%d] '%c' (0x%02x): %u\n",
               i + 1, best_next > 31 ? best_next : '?', best_next, best_count);

        visited[best_next] = 1;
    }
}
