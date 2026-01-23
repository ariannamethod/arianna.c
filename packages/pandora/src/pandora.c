// pandora.c — Release vocabulary from External Brain
// "Take the words, leave the voice"
//
// Pandora Package — Pure C vocabulary extraction using GPT2-30M
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "pandora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE PATTERNS (from Locus)
// ═══════════════════════════════════════════════════════════════════════════════

#define PATTERN_NONE          0
#define PATTERN_CRISIS        1
#define PATTERN_DISSOLUTION   2
#define PATTERN_EMERGENCE     3
#define PATTERN_TRANSCENDENCE 4

// ═══════════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_init(PandoraBox* pandora) {
    memset(pandora, 0, sizeof(PandoraBox));
    pandora->injection_strength = 0.2f;
    pandora->mode = PANDORA_MODE_AUTO;
    pandora->coherence_threshold = 0.3f;
    pandora->sacred_threshold = 0.7f;
}

int pandora_init_with_brain(
    PandoraBox* pandora,
    const char* weights_path,
    const char* vocab_path
) {
    pandora_init(pandora);

    // Allocate brain
    pandora->brain = malloc(sizeof(GPT2_30M));
    pandora->vocab = malloc(sizeof(GPT2Vocab));

    if (!pandora->brain || !pandora->vocab) {
        pandora_free(pandora);
        return -1;
    }

    pandora->brain_owned = 1;

    // Load weights
    if (gpt2_30m_load(pandora->brain, weights_path) != 0) {
        fprintf(stderr, "[pandora] failed to load brain weights\n");
        pandora_free(pandora);
        return -1;
    }

    // Load vocab
    if (gpt2_vocab_load(pandora->vocab, vocab_path) != 0) {
        fprintf(stderr, "[pandora] failed to load vocab\n");
        pandora_free(pandora);
        return -1;
    }

    fprintf(stderr, "[pandora] initialized with GPT2-30M brain\n");
    return 0;
}

void pandora_free(PandoraBox* pandora) {
    if (pandora->brain_owned) {
        if (pandora->brain) {
            gpt2_30m_free(pandora->brain);
            free(pandora->brain);
        }
        if (pandora->vocab) {
            gpt2_vocab_free(pandora->vocab);
            free(pandora->vocab);
        }
    }
    memset(pandora, 0, sizeof(PandoraBox));
}

// ═══════════════════════════════════════════════════════════════════════════════
// N-GRAM EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════

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

void pandora_extract(
    PandoraBox* pandora,
    const int* tokens,
    int n_tokens,
    int min_n,
    int max_n
) {
    if (pandora->mode == PANDORA_MODE_OFF) return;
    if (min_n < 1) min_n = 1;
    if (max_n > PANDORA_MAX_NGRAM_LEN) max_n = PANDORA_MAX_NGRAM_LEN;

    for (int n = min_n; n <= max_n; n++) {
        for (int start = 0; start <= n_tokens - n; start++) {
            const int* ngram = &tokens[start];

            int idx = find_ngram(pandora, ngram, n);

            if (idx >= 0) {
                pandora->ngrams[idx].frequency++;
                pandora->ngrams[idx].weight += 0.01f;
                if (pandora->ngrams[idx].weight > 1.0f) {
                    pandora->ngrams[idx].weight = 1.0f;
                }
            } else if (pandora->n_ngrams < PANDORA_MAX_NGRAMS) {
                ReleasedNGram* ng = &pandora->ngrams[pandora->n_ngrams];
                memcpy(ng->tokens, ngram, n * sizeof(int));
                ng->length = n;
                ng->weight = 0.1f;
                ng->frequency = 1;
                ng->arianna_mapped = 0;
                pandora->n_ngrams++;
                pandora->total_released++;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOCABULARY MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_map_to_arianna(
    PandoraBox* pandora,
    const char* (*brain_decode)(int token_id),
    int (*arianna_encode)(const char* word)
) {
    if (!brain_decode || !arianna_encode) return 0;

    int mapped = 0;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (ng->arianna_mapped) continue;

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
// LOGITS INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_apply_to_logits(
    PandoraBox* pandora,
    float* logits,
    const int* context_tokens,
    int context_len,
    int vocab_size
) {
    if (!pandora_is_active(pandora)) return;
    if (pandora->injection_strength <= 0.0f) return;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (!ng->arianna_mapped) continue;
        if (ng->frequency < PANDORA_MIN_FREQUENCY) continue;

        int prefix_len = ng->length - 1;
        if (prefix_len > context_len) continue;

        if (prefix_len == 0) {
            // Unigram boost
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
            int next_tok = ng->arianna_tokens[ng->length - 1];
            if (next_tok >= 0 && next_tok < vocab_size) {
                float boost = ng->weight * pandora->injection_strength;
                boost *= (1.0f + 0.1f * ng->frequency);
                logits[next_tok] += boost;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SARTRE INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_check_sartre(
    PandoraBox* pandora,
    float coherence,
    float sacred,
    int pattern
) {
    // Deactivate on high sacred (protect voice)
    if (sacred > pandora->sacred_threshold) {
        return 0;
    }

    // Deactivate on CRISIS (internal processing)
    if (pattern == PATTERN_CRISIS) {
        return 0;
    }

    // Activate on low coherence (need vocabulary boost)
    if (coherence < pandora->coherence_threshold) {
        return 1;
    }

    // Activate on EMERGENCE (creative expansion)
    if (pattern == PATTERN_EMERGENCE) {
        return 1;
    }

    // Activate on TRANSCENDENCE (bridging)
    if (pattern == PATTERN_TRANSCENDENCE) {
        return 1;
    }

    // Default: maintain current state
    return pandora->mode == PANDORA_MODE_FORCED ? 1 : 0;
}

void pandora_set_thresholds(
    PandoraBox* pandora,
    float coherence_threshold,
    float sacred_threshold
) {
    pandora->coherence_threshold = coherence_threshold;
    pandora->sacred_threshold = sacred_threshold;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

// Simple BPE-like encoding for external brain
static int simple_encode_for_brain(const char* text, int* tokens, int max_tokens) {
    // Very basic: just treat as characters for now
    // Real implementation would use BPE
    int n = 0;
    while (*text && n < max_tokens) {
        // Map ASCII to token range
        tokens[n++] = (unsigned char)*text;
        text++;
    }
    return n;
}

int pandora_process(
    PandoraBox* pandora,
    const char* input,
    int (*arianna_encode)(const char* word),
    int max_generate
) {
    if (!pandora->brain || !pandora->vocab) {
        fprintf(stderr, "[pandora] no brain loaded\n");
        return 0;
    }

    if (!pandora_is_active(pandora)) {
        return 0;
    }

    // Encode input
    int prompt_tokens[GPT2_30M_CONTEXT_LEN];
    int n_prompt = simple_encode_for_brain(input, prompt_tokens, GPT2_30M_CONTEXT_LEN / 2);

    // Generate from brain
    int output_tokens[GPT2_30M_CONTEXT_LEN];
    int n_output = gpt2_30m_generate(
        pandora->brain,
        prompt_tokens,
        n_prompt,
        output_tokens,
        max_generate,
        0.8f  // temperature
    );

    if (n_output <= 0) {
        return 0;
    }

    // Extract n-grams
    int prev_ngrams = pandora->n_ngrams;
    pandora_extract(pandora, output_tokens, n_output, 1, 3);

    // Map to Arianna vocab
    const char* (*decode_fn)(int) = (const char* (*)(int))gpt2_vocab_decode;
    // Note: This cast is for demonstration - real code would use proper wrapper
    pandora_map_to_arianna(pandora, decode_fn, arianna_encode);

    return pandora->n_ngrams - prev_ngrams;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODE & CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_set_mode(PandoraBox* pandora, PandoraMode mode) {
    pandora->mode = mode;
}

void pandora_set_strength(PandoraBox* pandora, float strength) {
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    pandora->injection_strength = strength;
}

int pandora_is_active(PandoraBox* pandora) {
    if (pandora->mode == PANDORA_MODE_OFF) return 0;
    if (pandora->mode == PANDORA_MODE_FORCED) return 1;
    // AUTO mode - controlled externally via pandora_check_sartre
    return 1;  // Assume active unless explicitly disabled
}

// ═══════════════════════════════════════════════════════════════════════════════
// DECAY
// ═══════════════════════════════════════════════════════════════════════════════

void pandora_decay(PandoraBox* pandora, float decay_rate) {
    float sum_weight = 0.0f;
    int active_count = 0;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        pandora->ngrams[i].weight *= decay_rate;

        if (pandora->ngrams[i].weight < 0.01f) {
            if (i < pandora->n_ngrams - 1) {
                pandora->ngrams[i] = pandora->ngrams[pandora->n_ngrams - 1];
                i--;
            }
            pandora->n_ngrams--;
        } else {
            sum_weight += pandora->ngrams[i].weight;
            active_count++;
        }
    }

    pandora->avg_weight = active_count > 0 ? sum_weight / active_count : 0.0f;
}

int pandora_suggest_continuation(
    PandoraBox* pandora,
    const int* context_tokens,
    int context_len
) {
    if (!pandora_is_active(pandora)) return -1;

    int best_token = -1;
    float best_score = 0.0f;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        if (!ng->arianna_mapped) continue;
        if (ng->length < 2) continue;

        int prefix_len = ng->length - 1;
        if (prefix_len > context_len) continue;

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
// PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_save(PandoraBox* pandora, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Header
    int magic = 0x50414E44;  // 'PAND'
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&pandora->n_ngrams, sizeof(int), 1, f);
    fwrite(&pandora->injection_strength, sizeof(float), 1, f);
    fwrite(&pandora->mode, sizeof(int), 1, f);

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

    // Keep brain reference if exists
    GPT2_30M* brain = pandora->brain;
    GPT2Vocab* vocab = pandora->vocab;
    int owned = pandora->brain_owned;

    pandora_init(pandora);

    pandora->brain = brain;
    pandora->vocab = vocab;
    pandora->brain_owned = owned;

    // Header
    int magic;
    fread(&magic, sizeof(int), 1, f);
    if (magic != 0x50414E44) {
        fprintf(stderr, "[pandora] invalid magic\n");
        fclose(f);
        return -1;
    }

    fread(&pandora->n_ngrams, sizeof(int), 1, f);
    fread(&pandora->injection_strength, sizeof(float), 1, f);
    fread(&pandora->mode, sizeof(int), 1, f);

    if (pandora->n_ngrams > PANDORA_MAX_NGRAMS) {
        pandora->n_ngrams = PANDORA_MAX_NGRAMS;
    }

    // N-grams
    for (int i = 0; i < pandora->n_ngrams; i++) {
        ReleasedNGram* ng = &pandora->ngrams[i];
        fread(&ng->length, sizeof(int), 1, f);
        fread(ng->tokens, sizeof(int), ng->length, f);
        fread(&ng->weight, sizeof(float), 1, f);
        fread(&ng->frequency, sizeof(int), 1, f);
        fread(&ng->arianna_mapped, sizeof(int), 1, f);
        if (ng->arianna_mapped) {
            fread(ng->arianna_tokens, sizeof(int), ng->length, f);
        }
    }

    fclose(f);
    fprintf(stderr, "[pandora] loaded %d n-grams from %s\n", pandora->n_ngrams, path);
    return 0;
}

void pandora_clear(PandoraBox* pandora) {
    pandora->n_ngrams = 0;
    pandora->total_released = 0;
    pandora->successfully_mapped = 0;
    pandora->avg_weight = 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATS
// ═══════════════════════════════════════════════════════════════════════════════

PandoraStats pandora_get_stats(PandoraBox* pandora) {
    PandoraStats stats = {0};

    stats.total_ngrams = pandora->n_ngrams;
    stats.mode = pandora->mode;
    stats.injection_strength = pandora->injection_strength;

    float sum_weight = 0.0f;
    float sum_freq = 0.0f;
    int mapped = 0;

    for (int i = 0; i < pandora->n_ngrams; i++) {
        sum_weight += pandora->ngrams[i].weight;
        sum_freq += pandora->ngrams[i].frequency;
        if (pandora->ngrams[i].arianna_mapped) mapped++;
    }

    stats.mapped_ngrams = mapped;
    if (pandora->n_ngrams > 0) {
        stats.avg_weight = sum_weight / pandora->n_ngrams;
        stats.avg_frequency = sum_freq / pandora->n_ngrams;
    }

    return stats;
}
