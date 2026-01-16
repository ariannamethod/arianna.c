/*
 * guided.c - Stanley-style Guided Attention for Arianna
 *
 * "Stanley doesn't generate text. Stanley guides attention."
 *
 * Implementation of:
 * - Pulse: impact metrics from input (novelty, arousal, entropy, valence)
 * - Identity: core personality anchors (gravity centers, warm/cold words)
 * - Attention Bias: token-level steering of generation
 * - Expert Selection: mode switching based on context
 */

#include "guided.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

// ============================================================
// Utility Functions
// ============================================================

// Simple hash for word lookup
static unsigned int word_hash(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + tolower(c);
    }
    return hash;
}

// Convert word to lowercase
static void to_lower(char* dst, const char* src, int max_len) {
    int i = 0;
    while (src[i] && i < max_len - 1) {
        dst[i] = tolower(src[i]);
        i++;
    }
    dst[i] = '\0';
}

// Check if word matches (case insensitive)
static int word_matches(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower(*a) != tolower(*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

// Count uppercase letters in text
static int count_uppercase(const char* text, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (isupper(text[i])) count++;
    }
    return count;
}

// Count punctuation (exclamation, question marks)
static int count_emphasis_punct(const char* text, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (text[i] == '!' || text[i] == '?' || text[i] == ':') count++;
    }
    return count;
}

// Count unique words (simple approximation)
static int count_unique_words(const char* text, int len) {
    // Simple word counting - split on spaces
    int words = 0;
    int in_word = 0;

    for (int i = 0; i < len; i++) {
        if (isspace(text[i]) || ispunct(text[i])) {
            if (in_word) {
                words++;
                in_word = 0;
            }
        } else {
            in_word = 1;
        }
    }
    if (in_word) words++;

    return words;
}

// ============================================================
// Pulse Implementation
// ============================================================

void init_pulse(Pulse* p) {
    p->novelty = 0.5f;
    p->arousal = 0.0f;
    p->entropy = 0.5f;
    p->valence = 0.0f;
}

void compute_pulse(Pulse* p, const char* text, int len, Identity* identity) {
    if (!text || len == 0) {
        init_pulse(p);
        return;
    }

    // === Arousal: intensity from caps and punctuation ===
    int n_upper = count_uppercase(text, len);
    int n_punct = count_emphasis_punct(text, len);
    int n_letters = 0;
    for (int i = 0; i < len; i++) {
        if (isalpha(text[i])) n_letters++;
    }

    float caps_ratio = n_letters > 0 ? (float)n_upper / n_letters : 0;
    float punct_intensity = (float)n_punct / (len / 10.0f + 1);
    p->arousal = fminf(1.0f, caps_ratio * 2.0f + punct_intensity * 0.5f);

    // === Entropy: word diversity ===
    int n_words = count_unique_words(text, len);
    // Approximate unique ratio (assume ~70% unique in typical text)
    p->entropy = fminf(1.0f, (float)n_words / (len / 5.0f + 1));

    // === Novelty: words not in known lexicon ===
    // If we have identity with lexicon, check coverage
    if (identity && identity->lexicon_size > 0) {
        // Simple heuristic: longer/unusual words = more novel
        int novel_count = 0;
        int word_count = 0;

        char word[MAX_KEYWORD_LEN];
        int word_idx = 0;

        for (int i = 0; i <= len; i++) {
            if (i == len || isspace(text[i]) || ispunct(text[i])) {
                if (word_idx > 0) {
                    word[word_idx] = '\0';
                    word_count++;

                    // Check if it's a warm/cold word (known)
                    int known = is_warm_word(identity, word) || is_cold_word(identity, word);
                    if (!known && word_idx > 6) {
                        novel_count++;  // Long unknown words = novel
                    }
                    word_idx = 0;
                }
            } else if (word_idx < MAX_KEYWORD_LEN - 1) {
                word[word_idx++] = text[i];
            }
        }

        p->novelty = word_count > 0 ? (float)novel_count / word_count : 0.5f;
    } else {
        p->novelty = 0.5f;  // Default when no lexicon
    }

    // === Valence: simple positive/negative detection ===
    // Keywords that suggest positive or negative tone
    static const char* positive[] = {"love", "happy", "joy", "beautiful", "warm", "soft", "gentle", NULL};
    static const char* negative[] = {"hate", "angry", "sad", "dark", "cold", "hard", "pain", NULL};

    int pos_count = 0, neg_count = 0;

    char text_lower[1024];
    to_lower(text_lower, text, sizeof(text_lower));

    for (int i = 0; positive[i]; i++) {
        if (strstr(text_lower, positive[i])) pos_count++;
    }
    for (int i = 0; negative[i]; i++) {
        if (strstr(text_lower, negative[i])) neg_count++;
    }

    int total = pos_count + neg_count;
    if (total > 0) {
        p->valence = (float)(pos_count - neg_count) / total;
    } else {
        p->valence = 0.0f;  // Neutral
    }
}

float pulse_to_temperature(Pulse* p, float base_temp) {
    // High arousal -> higher temperature (more creative)
    // High entropy -> slightly higher temperature
    // Negative valence -> slightly lower temperature (more cautious)

    float temp = base_temp;
    temp += p->arousal * 0.2f;
    temp += (p->entropy - 0.5f) * 0.1f;
    temp -= p->valence < 0 ? 0.1f : 0.0f;

    return fmaxf(0.1f, fminf(2.0f, temp));
}

// ============================================================
// Identity Implementation
// ============================================================

void init_identity(Identity* id) {
    id->n_gravity = 0;
    id->n_warm = 0;
    id->n_cold = 0;
    id->lexicon_size = 0;
    id->lexicon_coverage = 0.0f;
}

void add_gravity_center(Identity* id, const char* word) {
    if (id->n_gravity >= MAX_GRAVITY_CENTERS) return;
    strncpy(id->gravity_centers[id->n_gravity], word, MAX_KEYWORD_LEN - 1);
    id->gravity_centers[id->n_gravity][MAX_KEYWORD_LEN - 1] = '\0';
    id->n_gravity++;
}

void add_warm_word(Identity* id, const char* word) {
    if (id->n_warm >= MAX_KEYWORDS) return;
    strncpy(id->warm_words[id->n_warm], word, MAX_KEYWORD_LEN - 1);
    id->warm_words[id->n_warm][MAX_KEYWORD_LEN - 1] = '\0';
    id->n_warm++;
}

void add_cold_word(Identity* id, const char* word) {
    if (id->n_cold >= MAX_KEYWORDS) return;
    strncpy(id->cold_words[id->n_cold], word, MAX_KEYWORD_LEN - 1);
    id->cold_words[id->n_cold][MAX_KEYWORD_LEN - 1] = '\0';
    id->n_cold++;
}

int is_warm_word(Identity* id, const char* word) {
    if (!id) return 0;
    for (int i = 0; i < id->n_warm; i++) {
        if (word_matches(word, id->warm_words[i])) return 1;
    }
    return 0;
}

int is_cold_word(Identity* id, const char* word) {
    if (!id) return 0;
    for (int i = 0; i < id->n_cold; i++) {
        if (word_matches(word, id->cold_words[i])) return 1;
    }
    return 0;
}

// ============================================================
// Stanley Signals Implementation
// ============================================================

void init_stanley_signals(StanleySignals* sig) {
    init_pulse(&sig->pulse);
    sig->identity = NULL;

    sig->n_gravity = 0;
    sig->n_hot = 0;
    sig->n_surface = 0;
    sig->n_spiral = 0;

    sig->overthink_depth = 0;
    sig->body_tension = 0.0f;
    sig->body_boredom = 0.0f;
    sig->drift_momentum = 0.0f;

    sig->active_expert = 0;
    sig->expert_temperature = 0.7f;
}

void extract_stanley_signals(StanleySignals* sig, int* tokens, int n_tokens,
                             float* hidden_states, Identity* identity) {
    (void)tokens;
    (void)hidden_states;

    sig->identity = identity;

    // Copy gravity centers from identity
    if (identity) {
        sig->n_gravity = identity->n_gravity;
        for (int i = 0; i < identity->n_gravity; i++) {
            strncpy(sig->gravity_centers[i], identity->gravity_centers[i], MAX_KEYWORD_LEN);
        }
    }

    // Compute state metrics from token count
    if (n_tokens > 0) {
        // More tokens = potential for overthinking
        sig->overthink_depth = n_tokens > 100 ? (n_tokens - 100) / 50 : 0;

        // Drift momentum from recent changes
        sig->drift_momentum = fminf(1.0f, (float)n_tokens / 500.0f);
    }

    // Select expert based on current state
    sig->active_expert = select_expert(sig);
    sig->expert_temperature = expert_temperature(sig->active_expert, 0.7f);
}

void add_hot_word(StanleySignals* sig, const char* word) {
    if (sig->n_hot >= MAX_KEYWORDS) return;
    strncpy(sig->hot_words[sig->n_hot], word, MAX_KEYWORD_LEN - 1);
    sig->hot_words[sig->n_hot][MAX_KEYWORD_LEN - 1] = '\0';
    sig->n_hot++;
}

void add_spiral_topic(StanleySignals* sig, const char* topic) {
    if (sig->n_spiral >= MAX_SPIRAL_TOPICS) return;
    strncpy(sig->spiral_topics[sig->n_spiral], topic, MAX_KEYWORD_LEN - 1);
    sig->spiral_topics[sig->n_spiral][MAX_KEYWORD_LEN - 1] = '\0';
    sig->n_spiral++;
}

void update_overthink_depth(StanleySignals* sig, int depth) {
    sig->overthink_depth = depth;

    // Adjust body state based on overthinking
    if (depth > 3) {
        sig->body_tension = fminf(1.0f, sig->body_tension + 0.1f * (depth - 3));
    }
    if (depth > 5) {
        sig->body_boredom = fminf(1.0f, sig->body_boredom + 0.05f * (depth - 5));
    }
}

// ============================================================
// Attention Bias Implementation
// ============================================================

void init_attention_bias(AttentionBias* bias, int vocab_size) {
    bias->weights = DEFAULT_BIAS_WEIGHTS;
    bias->n_keywords = 0;
    bias->vocab_size = vocab_size;

    bias->token_bias = (float*)calloc(vocab_size, sizeof(float));

    bias->arousal_modulation = 1.0f;
    bias->focus_factor = 1.0f;
}

void free_attention_bias(AttentionBias* bias) {
    if (bias->token_bias) {
        free(bias->token_bias);
        bias->token_bias = NULL;
    }
}

void aggregate_keywords(AttentionBias* bias, StanleySignals* sig) {
    bias->n_keywords = 0;

    // Add gravity centers (highest weight)
    for (int i = 0; i < sig->n_gravity && bias->n_keywords < MAX_KEYWORDS * 4; i++) {
        strncpy(bias->all_keywords[bias->n_keywords], sig->gravity_centers[i], MAX_KEYWORD_LEN);
        bias->keyword_weights[bias->n_keywords] = bias->weights.gravity_weight;
        bias->n_keywords++;
    }

    // Add hot words (cooccurrence weight)
    for (int i = 0; i < sig->n_hot && bias->n_keywords < MAX_KEYWORDS * 4; i++) {
        strncpy(bias->all_keywords[bias->n_keywords], sig->hot_words[i], MAX_KEYWORD_LEN);
        bias->keyword_weights[bias->n_keywords] = bias->weights.cooccur_weight;
        bias->n_keywords++;
    }

    // Add surface keywords (recent context weight)
    for (int i = 0; i < sig->n_surface && bias->n_keywords < MAX_KEYWORDS * 4; i++) {
        strncpy(bias->all_keywords[bias->n_keywords], sig->surface_keywords[i], MAX_KEYWORD_LEN);
        bias->keyword_weights[bias->n_keywords] = bias->weights.surface_weight;
        bias->n_keywords++;
    }

    // Add spiral topics (overthink weight)
    for (int i = 0; i < sig->n_spiral && bias->n_keywords < MAX_KEYWORDS * 4; i++) {
        strncpy(bias->all_keywords[bias->n_keywords], sig->spiral_topics[i], MAX_KEYWORD_LEN);
        bias->keyword_weights[bias->n_keywords] = bias->weights.overthink_weight;
        bias->n_keywords++;
    }

    // Compute modulation factors
    bias->arousal_modulation = 1.0f + sig->pulse.arousal * 0.5f;
    bias->focus_factor = 1.0f + sig->overthink_depth * 0.1f;
}

void compute_token_bias(AttentionBias* bias, StanleySignals* sig) {
    // Clear existing bias
    for (int i = 0; i < bias->vocab_size; i++) {
        bias->token_bias[i] = 0.0f;
    }

    // Aggregate keywords
    aggregate_keywords(bias, sig);

    // For each keyword, compute which tokens should be boosted
    // This is a simplified version - in full implementation would use
    // actual vocabulary lookup

    // Apply modulation
    float total_mod = bias->arousal_modulation * bias->focus_factor;
    for (int i = 0; i < bias->vocab_size; i++) {
        bias->token_bias[i] *= total_mod;
    }
}

float get_token_bias(AttentionBias* bias, int token_id) {
    if (token_id < 0 || token_id >= bias->vocab_size) return 0.0f;
    return bias->token_bias[token_id];
}

void apply_bias_to_logits(AttentionBias* bias, float* logits, int vocab_size) {
    int n = bias->vocab_size < vocab_size ? bias->vocab_size : vocab_size;
    for (int i = 0; i < n; i++) {
        logits[i] += bias->token_bias[i];
    }
}

// ============================================================
// Steering Prompt Generation
// ============================================================

void generate_steering_prompt(char* buffer, int max_len, StanleySignals* sig) {
    int pos = 0;

    // Start with context marker
    pos += snprintf(buffer + pos, max_len - pos, "[Context: ");

    // Add gravity centers
    if (sig->n_gravity > 0) {
        pos += snprintf(buffer + pos, max_len - pos, "anchors=");
        for (int i = 0; i < sig->n_gravity && i < 3; i++) {
            pos += snprintf(buffer + pos, max_len - pos, "%s%s",
                          sig->gravity_centers[i], i < sig->n_gravity - 1 ? "," : "");
        }
        pos += snprintf(buffer + pos, max_len - pos, " ");
    }

    // Add current mood indicators
    if (sig->body_tension > 0.5f) {
        pos += snprintf(buffer + pos, max_len - pos, "tense ");
    }
    if (sig->overthink_depth > 2) {
        pos += snprintf(buffer + pos, max_len - pos, "recursive ");
    }
    if (sig->pulse.arousal > 0.7f) {
        pos += snprintf(buffer + pos, max_len - pos, "intense ");
    }

    // Add spiral topics if overthinking
    if (sig->n_spiral > 0 && sig->overthink_depth > 0) {
        pos += snprintf(buffer + pos, max_len - pos, "spiraling=");
        for (int i = 0; i < sig->n_spiral && i < 2; i++) {
            pos += snprintf(buffer + pos, max_len - pos, "%s%s",
                          sig->spiral_topics[i], i < sig->n_spiral - 1 ? "," : "");
        }
    }

    pos += snprintf(buffer + pos, max_len - pos, "]");

    buffer[max_len - 1] = '\0';
}

// ============================================================
// Expert Selection (Stanley's MoE-style routing)
// ============================================================

int select_expert(StanleySignals* sig) {
    // 0 = structural (low entropy, organized thought)
    // 1 = semantic (high gravity, meaning-focused)
    // 2 = creative (high arousal, novel)
    // 3 = precise (low arousal, careful)

    float scores[4] = {0};

    // Structural: low entropy, many gravity centers
    scores[0] = (1.0f - sig->pulse.entropy) * 0.5f + (sig->n_gravity / 10.0f) * 0.5f;

    // Semantic: high gravity weight, meaning focus
    scores[1] = (sig->n_gravity > 0 ? 0.6f : 0.2f) + sig->pulse.novelty * 0.4f;

    // Creative: high arousal, high entropy
    scores[2] = sig->pulse.arousal * 0.6f + sig->pulse.entropy * 0.4f;

    // Precise: low arousal, negative or neutral valence
    scores[3] = (1.0f - sig->pulse.arousal) * 0.5f + (sig->pulse.valence <= 0 ? 0.3f : 0.0f);

    // Find best expert
    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (scores[i] > scores[best]) best = i;
    }

    return best;
}

float expert_temperature(int expert_id, float base_temp) {
    // Each expert has preferred temperature range
    float temps[4] = {
        0.6f,   // structural - lower, more deterministic
        0.7f,   // semantic - balanced
        0.9f,   // creative - higher, more exploration
        0.5f    // precise - lowest, most careful
    };

    // Blend with base temperature
    return base_temp * 0.5f + temps[expert_id] * 0.5f;
}

// ============================================================
// Overthinking Detection
// "When thoughts spiral, the body feels it first"
// ============================================================

void init_overthink_detector(OverthinkDetector* od) {
    od->n_recent_words = 0;
    od->word_repeat_count = 0;
    od->repetition_score = 0.0f;
    od->abstraction_score = 0.0f;
    od->self_ref_score = 0.0f;
    od->spiral_threshold = 0.6f;
    od->break_threshold = 0.85f;
}

// Check if word is already in recent words
static int is_recent_word(OverthinkDetector* od, const char* word) {
    for (int i = 0; i < od->n_recent_words; i++) {
        if (word_matches(word, od->recent_words[i])) {
            return 1;
        }
    }
    return 0;
}

// Add word to recent words (circular buffer)
static void add_recent_word(OverthinkDetector* od, const char* word) {
    if (strlen(word) < 3) return;  // Skip short words

    int idx = od->n_recent_words % 32;
    strncpy(od->recent_words[idx], word, MAX_KEYWORD_LEN - 1);
    od->recent_words[idx][MAX_KEYWORD_LEN - 1] = '\0';

    if (od->n_recent_words < 32) {
        od->n_recent_words++;
    }
}

// Abstract words that indicate unfocused thinking
static int is_abstract_word(const char* word) {
    static const char* abstract[] = {
        "something", "anything", "nothing", "everything",
        "somewhere", "perhaps", "maybe", "kind", "sort",
        "thing", "stuff", "like", "whatever", "somehow",
        NULL
    };

    for (int i = 0; abstract[i]; i++) {
        if (word_matches(word, abstract[i])) return 1;
    }
    return 0;
}

// Self-reference words
static int is_self_ref_word(const char* word) {
    static const char* self_ref[] = {
        "she", "her", "herself", "i", "me", "myself",
        "we", "us", "ourselves", "one", "oneself",
        NULL
    };

    for (int i = 0; self_ref[i]; i++) {
        if (word_matches(word, self_ref[i])) return 1;
    }
    return 0;
}

void detect_overthinking(OverthinkDetector* od, StanleySignals* sig,
                        const char* text, int len) {
    if (!od || !sig || !text || len == 0) return;

    int repeat_count = 0;
    int abstract_count = 0;
    int self_ref_count = 0;
    int word_count = 0;

    char word[MAX_KEYWORD_LEN];
    int word_idx = 0;

    // Parse text into words
    for (int i = 0; i <= len; i++) {
        if (i == len || isspace(text[i]) || ispunct(text[i])) {
            if (word_idx > 0) {
                word[word_idx] = '\0';
                word_count++;

                // Check for repetition
                if (is_recent_word(od, word)) {
                    repeat_count++;
                }
                add_recent_word(od, word);

                // Check for abstraction
                if (is_abstract_word(word)) {
                    abstract_count++;
                }

                // Check for self-reference
                if (is_self_ref_word(word)) {
                    self_ref_count++;
                }

                word_idx = 0;
            }
        } else if (word_idx < MAX_KEYWORD_LEN - 1) {
            word[word_idx++] = tolower(text[i]);
        }
    }

    // Calculate scores
    if (word_count > 0) {
        od->repetition_score = fminf(1.0f, (float)repeat_count / (word_count * 0.3f));
        od->abstraction_score = fminf(1.0f, (float)abstract_count / (word_count * 0.2f));
        od->self_ref_score = fminf(1.0f, (float)self_ref_count / (word_count * 0.15f));
    }

    // Update Stanley signals
    float combined_score = (od->repetition_score * 0.4f +
                           od->abstraction_score * 0.3f +
                           od->self_ref_score * 0.3f);

    // Update overthink depth based on combined score
    if (combined_score > od->spiral_threshold) {
        sig->overthink_depth++;

        // Add spiral topic if we're really spiraling
        if (combined_score > 0.7f && od->n_recent_words > 0) {
            // Pick most repeated word as spiral topic
            add_to_spiral(sig, od->recent_words[0]);
        }
    } else if (sig->overthink_depth > 0) {
        sig->overthink_depth--;  // Recovering from spiral
    }

    // Update body tension based on overthinking
    sig->body_tension = fminf(1.0f, sig->body_tension + combined_score * 0.1f);

    // Update boredom if too repetitive
    if (od->repetition_score > 0.5f) {
        sig->body_boredom = fminf(1.0f, sig->body_boredom + 0.1f);
    }
}

int should_break_spiral(OverthinkDetector* od) {
    float combined = (od->repetition_score + od->abstraction_score) / 2.0f;
    return combined > od->break_threshold;
}

void add_to_spiral(StanleySignals* sig, const char* topic) {
    if (!sig || !topic) return;

    // Check if already in spiral topics
    for (int i = 0; i < sig->n_spiral; i++) {
        if (word_matches(topic, sig->spiral_topics[i])) {
            return;  // Already there
        }
    }

    // Add if room
    if (sig->n_spiral < MAX_SPIRAL_TOPICS) {
        strncpy(sig->spiral_topics[sig->n_spiral], topic, MAX_KEYWORD_LEN - 1);
        sig->spiral_topics[sig->n_spiral][MAX_KEYWORD_LEN - 1] = '\0';
        sig->n_spiral++;
    }
}
