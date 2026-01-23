// pandora.c — Release vocabulary from External Brain
// "Take the words, leave the voice"
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "pandora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_init(PandoraBox* pandora) {
    memset(pandora, 0, sizeof(PandoraBox));
    pandora->injection_strength = 0.2f;  // subtle by default
    pandora->active = 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTRACT N-GRAMS
// ═══════════════════════════════════════════════════════════════════════════════

// Hash function for n-gram lookup
static unsigned int hash_ngram(const int* tokens, int len) {
    unsigned int h = 0;
    for (int i = 0; i < len; i++) {
        h = h * 31 + (unsigned int)tokens[i];
    }
    return h;
}

// Find existing n-gram or return -1
static int find_ngram(PandoraBox* pandora, const int* tokens, int len) {
    for (int i = 0; i < pandora->n_ngrams; i++) {
        if (pandora->ngrams[i].length != len) continue;

        int match = 1;
        for (int j = 0; j < len; j++) {
            if (pandora->ngrams[i].tokens[j] != tokens[j]) {
                match = 0;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
}

void pandora_extract(PandoraBox* pandora,
                   const int* tokens, int n_tokens,
                   int min_n, int max_n) {
    if (!pandora->active) return;
    if (min_n < 1) min_n = 1;
    if (max_n > PANDORA_MAX_NGRAM_LEN) max_n = PANDORA_MAX_NGRAM_LEN;

    // Extract all n-grams of length min_n to max_n
    for (int n = min_n; n <= max_n; n++) {
        for (int start = 0; start <= n_tokens - n; start++) {
            const int* ngram = &tokens[start];

            // Check if already exists
            int idx = find_ngram(pandora, ngram, n);

            if (idx >= 0) {
                // Increment frequency
                pandora->ngrams[idx].frequency++;
                // Boost weight slightly
                pandora->ngrams[idx].weight += 0.01f;
                if (pandora->ngrams[idx].weight > 1.0f) {
                    pandora->ngrams[idx].weight = 1.0f;
                }
            } else if (pandora->n_ngrams < PANDORA_MAX_NGRAMS) {
                // Add new n-gram
                ReleasedNGram* ng = &pandora->ngrams[pandora->n_ngrams];
                memcpy(ng->tokens, ngram, n * sizeof(int));
                ng->length = n;
                ng->weight = 0.1f;  // start low
                ng->frequency = 1;
                ng->arianna_mapped = 0;
                pandora->n_ngrams++;
                pandora->total_released++;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAP TO ARIANNA VOCABULARY
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_map_to_arianna(PandoraBox* pandora,
                         const char* (*brain_decode)(int token_id),
                         int (*arianna_encode)(const char* word)) {
    if (!brain_decode || !arianna_encode) return 0;

    int mapped = 0;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (ng->arianna_mapped) continue;

        // Decode each token and re-encode for Arianna
        int success = 1;
        for (int j = 0; j < ng->length; j++) {
            const char* word = brain_decode(ng->tokens[j]);
            if (!word) {
                success = 0;
                break;
            }

            int arianna_id = arianna_encode(word);
            if (arianna_id < 0) {
                success = 0;
                break;
            }

            ng->arianna_tokens[j] = arianna_id;
        }

        if (success) {
            ng->arianna_mapped = 1;
            mapped++;
        }
    }

    pandora->successfully_mapped = mapped;
    return mapped;
}

// ═══════════════════════════════════════════════════════════════════════════════
// APPLY TO LOGITS
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_apply_to_logits(PandoraBox* pandora,
                           float* logits,
                           const int* context_tokens,
                           int context_len,
                           int vocab_size) {
    if (!pandora->active || pandora->injection_strength <= 0.0f) return;

    // For each released n-gram, check if context ends with its prefix
    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (!ng->arianna_mapped) continue;
        if (ng->frequency < PANDORA_MIN_FREQUENCY) continue;

        // Check if context ends with n-gram prefix (all but last token)
        int prefix_len = ng->length - 1;
        if (prefix_len > context_len) continue;
        if (prefix_len == 0) {
            // Unigram - always boost slightly
            int tok = ng->arianna_tokens[0];
            if (tok >= 0 && tok < vocab_size) {
                logits[tok] += ng->weight * pandora->injection_strength * 0.5f;
            }
            continue;
        }

        // Check prefix match
        int match = 1;
        for (int j = 0; j < prefix_len; j++) {
            int ctx_idx = context_len - prefix_len + j;
            if (context_tokens[ctx_idx] != ng->arianna_tokens[j]) {
                match = 0;
                break;
            }
        }

        if (match) {
            // Boost the continuation token
            int next_tok = ng->arianna_tokens[ng->length - 1];
            if (next_tok >= 0 && next_tok < vocab_size) {
                float boost = ng->weight * pandora->injection_strength;
                // Scale by frequency (more frequent = stronger)
                boost *= (1.0f + 0.1f * ng->frequency);
                logits[next_tok] += boost;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUGGEST CONTINUATION
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_suggest_continuation(PandoraBox* pandora,
                               const int* context_tokens,
                               int context_len) {
    if (!pandora->active) return -1;

    int best_token = -1;
    float best_score = 0.0f;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (!ng->arianna_mapped) continue;
        if (ng->length < 2) continue;

        int prefix_len = ng->length - 1;
        if (prefix_len > context_len) continue;

        // Check prefix match
        int match = 1;
        for (int j = 0; j < prefix_len; j++) {
            int ctx_idx = context_len - prefix_len + j;
            if (context_tokens[ctx_idx] != ng->arianna_tokens[j]) {
                match = 0;
                break;
            }
        }

        if (match) {
            float score = ng->weight * ng->frequency;
            if (score > best_score) {
                best_score = score;
                best_token = ng->arianna_tokens[ng->length - 1];
            }
        }
    }

    return best_token;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECAY
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_decay(PandoraBox* pandora, float decay_rate) {
    float sum_weight = 0.0f;
    int active_count = 0;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        pandora->ngrams[i].weight *= decay_rate;

        // Remove if weight too low
        if (pandora->ngrams[i].weight < 0.01f) {
            // Swap with last and decrement
            if (i < pandora->n_ngrams - 1) {
                pandora->ngrams[i] = pandora->ngrams[pandora->n_ngrams - 1];
                i--;  // recheck this slot
            }
            pandora->n_ngrams--;
        } else {
            sum_weight += pandora->ngrams[i].weight;
            active_count++;
        }
    }

    pandora->avg_weight = active_count > 0 ? sum_weight / active_count : 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAVE / LOAD
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_save(PandoraBox* pandora, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Header
    fwrite(&pandora->n_ngrams, sizeof(int), 1, f);
    fwrite(&pandora->injection_strength, sizeof(float), 1, f);

    // N-grams
    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        fwrite(&ng->length, sizeof(int), 1, f);
        fwrite(ng->tokens, sizeof(int), ng->length, f);
        fwrite(&ng->weight, sizeof(float), 1, f);
        fwrite(&ng->frequency, sizeof(int), 1, f);
        fwrite(&ng->arianna_mapped, sizeof(int), 1, f);
        if (ng->arianna_mapped) {
            fwrite(ng->arianna_tokens, sizeof(int), ng->length, f);
        }
    }

    fclose(f);
    fprintf(stderr, "[pandora] saved %d n-grams to %s\n", pandora->n_ngrams, path);
    return 0;
}

int pandora_load(PandoraBox* pandora, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    pandora_init(pandora);

    // Header
    fread(&pandora->n_ngrams, sizeof(int), 1, f);
    fread(&pandora->injection_strength, sizeof(float), 1, f);

    if (pandora->n_ngrams > PANDORA_MAX_NGRAMS) {
        pandora->n_ngrams = PANDORA_MAX_NGRAMS;
    }

    // N-grams
    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (fread(&ng->length, sizeof(int), 1, f) != 1) {
            fclose(f);
            fprintf(stderr, "[pandora] corrupt file: failed to read n-gram length\n");
            return -1;
        }
        // SECURITY: validate length to prevent OOB write
        if (ng->length < 1 || ng->length > PANDORA_MAX_NGRAM_LEN) {
            fclose(f);
            fprintf(stderr, "[pandora] corrupt file: invalid n-gram length %d\n", ng->length);
            return -1;
        }
        if (fread(ng->tokens, sizeof(int), ng->length, f) != (size_t)ng->length) {
            fclose(f);
            return -1;
        }
        fread(&ng->weight, sizeof(float), 1, f);
        fread(&ng->frequency, sizeof(int), 1, f);
        fread(&ng->arianna_mapped, sizeof(int), 1, f);
        if (ng->arianna_mapped) {
            if (fread(ng->arianna_tokens, sizeof(int), ng->length, f) != (size_t)ng->length) {
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    fprintf(stderr, "[pandora] loaded %d n-grams from %s\n", pandora->n_ngrams, path);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETTERS
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_set_active(PandoraBox* pandora, int active) {
    pandora->active = active ? 1 : 0;
}

void pandora_set_strength(PandoraBox* pandora, float strength) {
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    pandora->injection_strength = strength;
}
