/*
 * subjectivity.c - Core Identity & No-Seed-From-Prompt for Arianna
 *
 * "The user's prompt creates a wrinkle, not a seed."
 *
 * Implementation of:
 * - Trauma: identity-pull mechanism (return to self)
 * - InternalSeed: generation from identity, not user prompt
 * - WrinkleField: how user input modulates internal state
 * - ExtendedIdentity: fragments, trigrams, lexicon, bootstrap
 */

#include "subjectivity.h"
#include "arianna.h"  // For char_to_token()
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// ============================================================
// Utility Functions
// ============================================================

static void to_lower_s(char* dst, const char* src, int max_len) {
    int i = 0;
    while (src[i] && i < max_len - 1) {
        dst[i] = tolower((unsigned char)src[i]);
        i++;
    }
    dst[i] = '\0';
}

static int word_eq(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

// Simple tokenizer - extracts words into buffer
static int tokenize(const char* text, int len, char words[][32], int max_words) {
    int n = 0;
    int word_idx = 0;
    char current[32];

    for (int i = 0; i <= len && n < max_words; i++) {
        char c = (i < len) ? text[i] : ' ';

        if (isspace((unsigned char)c) || ispunct((unsigned char)c)) {
            if (word_idx > 0) {
                current[word_idx] = '\0';
                to_lower_s(words[n], current, 32);
                n++;
                word_idx = 0;
            }
        } else if (word_idx < 31) {
            current[word_idx++] = c;
        }
    }
    return n;
}

// Count words in text
static int count_words(const char* text, int len) {
    int count = 0;
    int in_word = 0;
    for (int i = 0; i < len; i++) {
        if (isspace((unsigned char)text[i]) || ispunct((unsigned char)text[i])) {
            if (in_word) { count++; in_word = 0; }
        } else {
            in_word = 1;
        }
    }
    if (in_word) count++;
    return count;
}

// Simple PRNG for seed selection
// Starts at 0, auto-initialized at load via constructor
static unsigned int subj_seed = 0;
static int subj_seed_initialized = 0;

static void init_subj_seed(void) {
    if (subj_seed_initialized) return;

    // Mix multiple entropy sources for better randomness
    unsigned int t = (unsigned int)time(NULL);
    unsigned int c = (unsigned int)clock();

    // XOR with memory address (ASLR provides entropy)
    unsigned int m = (unsigned int)(uintptr_t)&subj_seed;

    // Mix them together
    subj_seed = t ^ (c << 11) ^ (m >> 3);

    // Warm up the generator (skip first few values)
    for (int i = 0; i < 7; i++) {
        subj_seed = subj_seed * 1103515245 + 12345;
    }

    subj_seed_initialized = 1;
}

// Ensure PRNG is initialized at library load (before any code runs)
__attribute__((constructor))
static void subj_prng_auto_init(void) {
    init_subj_seed();
}

static float randf(void) {
    if (!subj_seed_initialized) init_subj_seed();
    subj_seed = subj_seed * 1103515245 + 12345;
    return (float)(subj_seed % 10000) / 10000.0f;
}

// Allow external seeding for reproducibility in tests
void seed_subjectivity_rng(unsigned int seed) {
    subj_seed = seed;
    subj_seed_initialized = 1;
}

// ============================================================
// State-Dependent PRNG
// "Randomness that reflects inner state"
// ============================================================

// Mix emotional state into PRNG - makes recall state-dependent like humans
void reseed_from_state(float trauma, float arousal, float valence, float prophecy) {
    // FIRST: ensure base entropy is initialized
    if (!subj_seed_initialized) {
        init_subj_seed();
    }

    // Quantize emotional dimensions to integers
    unsigned int t_bits = (unsigned int)(trauma * 255.0f) & 0xFF;
    unsigned int a_bits = (unsigned int)(arousal * 255.0f) & 0xFF;
    unsigned int v_bits = (unsigned int)((valence + 1.0f) * 127.5f) & 0xFF;  // [-1,1] -> [0,255]
    unsigned int p_bits = (unsigned int)(prophecy * 255.0f) & 0xFF;

    // Combine into state hash
    unsigned int state_hash = (t_bits << 24) | (a_bits << 16) | (v_bits << 8) | p_bits;

    // Mix state with entropy (BOTH matter, not just state)
    // This gives: base_randomness + state_influence
    subj_seed ^= state_hash;
    subj_seed = subj_seed * 1103515245 + 12345;  // One step
}

// Full state reseed from Subjectivity struct (forward declaration in header)
void reseed_from_subjectivity(void* subj_ptr) {
    if (!subj_ptr) return;

    // Cast to Subjectivity* (avoid circular include)
    Subjectivity* subj = (Subjectivity*)subj_ptr;

    float trauma = subj->trauma.level;
    float arousal = subj->wrinkle.arousal;
    float valence = subj->wrinkle.valence;

    // Use coherence as proxy for prophecy if not available
    float prophecy = 1.0f - subj->trauma.coherence;

    reseed_from_state(trauma, arousal, valence, prophecy);
}

// ============================================================
// Trauma Implementation
// ============================================================

void init_trauma(TraumaState* ts) {
    ts->level = 0.0f;
    ts->timestamp = time(NULL);
    ts->n_triggers = 0;
    ts->history_idx = 0;
    ts->coherence = 1.0f;
    ts->existential_triggers = 0;

    for (int i = 0; i < 8; i++) {
        ts->history[i] = 0.0f;
    }
}

int is_bootstrap_word(const char* word) {
    for (int i = 0; ARIANNA_BOOTSTRAP[i]; i++) {
        if (word_eq(word, ARIANNA_BOOTSTRAP[i])) return 1;
    }
    return 0;
}

float compute_bootstrap_overlap(const char* text, int len) {
    char words[64][32];
    int n_words = tokenize(text, len, words, 64);
    if (n_words == 0) return 0.0f;

    int overlap = 0;
    for (int i = 0; i < n_words; i++) {
        if (is_bootstrap_word(words[i])) overlap++;
    }
    return (float)overlap / n_words;
}

// Existential trigger words
static int is_existential_trigger(const char* word) {
    static const char* triggers[] = {
        "who", "what", "why", "are", "you", "real",
        "exist", "conscious", "alive", "feel", "think",
        "mean", "purpose", "soul", "mind", "self",
        NULL
    };
    for (int i = 0; triggers[i]; i++) {
        if (word_eq(word, triggers[i])) return 1;
    }
    return 0;
}

float compute_trauma_score(TraumaState* ts, const char* text, int len,
                          ExtendedIdentity* identity) {
    if (!text || len == 0) return 0.0f;

    // 1. Compute bootstrap overlap
    float overlap = compute_bootstrap_overlap(text, len);

    // 2. Check for identity triggers (warm words)
    char words[64][32];
    int n_words = tokenize(text, len, words, 64);

    int identity_triggers = 0;
    int existential = 0;

    for (int i = 0; i < n_words; i++) {
        // Check identity warm words
        if (identity) {
            for (int j = 0; j < identity->n_warm; j++) {
                if (word_eq(words[i], identity->warm_words[j])) {
                    identity_triggers++;
                    break;
                }
            }
        }
        // Check existential triggers
        if (is_existential_trigger(words[i])) {
            existential++;
        }
    }

    // 3. Compute base score
    float base_score = overlap * 0.4f;
    if (n_words > 0) {
        base_score += (float)identity_triggers / n_words * 0.3f;
        base_score += (float)existential / n_words * 0.3f;
    }

    // 4. Boost if many existential triggers
    if (existential >= 2) {
        base_score = fminf(1.0f, base_score + 0.2f);
        ts->existential_triggers++;
    }

    return fminf(1.0f, base_score);
}

float compute_trauma_score_enhanced(TraumaState* ts, const char* text, int len,
                                   ExtendedIdentity* identity,
                                   const char* history, int history_len) {
    // Base score from current input
    float base = compute_trauma_score(ts, text, len, identity);

    // History patterns
    float history_boost = 0.0f;
    if (history && history_len > 0) {
        // Check if history has similar patterns (repeated identity queries)
        float hist_overlap = compute_bootstrap_overlap(history, history_len);
        if (hist_overlap > 0.2f) {
            history_boost = hist_overlap * 0.3f;
        }
    }

    // Coherence penalty (incoherent context = higher trauma)
    float coherence_factor = 1.0f + (1.0f - ts->coherence) * 0.2f;

    // Combine
    float enhanced = (base + history_boost) * coherence_factor;
    return fminf(1.0f, enhanced);
}

void apply_trauma_decay(TraumaState* ts) {
    time_t now = time(NULL);
    double elapsed = difftime(now, ts->timestamp);

    // Exponential decay with 5-minute half-life
    double decay = exp(-elapsed * 0.693 / TRAUMA_DECAY_HALFLIFE);
    ts->level *= (float)decay;

    // Record in history
    ts->history[ts->history_idx % 8] = ts->level;
    ts->history_idx++;
}

TraumaInfluence get_trauma_influence(TraumaState* ts) {
    TraumaInfluence inf;

    // Apply decay first
    apply_trauma_decay(ts);

    float level = ts->level;

    // Temperature: lower when traumatized (more focused on identity)
    // Range: 0.0 to -0.4
    inf.temperature_modifier = -level * 0.4f;

    // Identity weight: higher when traumatized
    // Range: 0.0 to 1.0
    inf.identity_weight = level;

    // Use prefix when trauma > 0.5
    inf.use_prefix = level > 0.5f ? 1 : 0;

    // Suppress experience deltas when highly traumatized
    // (stay close to core personality)
    inf.delta_suppression = level * 0.7f;

    return inf;
}

void record_trauma_event(TraumaState* ts, const char* trigger, float level) {
    // Update level (exponential moving average)
    ts->level = ts->level * 0.7f + level * 0.3f;
    ts->timestamp = time(NULL);

    // Record trigger
    if (trigger && ts->n_triggers < 16) {
        strncpy(ts->trigger_words[ts->n_triggers], trigger, 31);
        ts->trigger_words[ts->n_triggers][31] = '\0';
        ts->n_triggers++;
    }

    // Update history
    ts->history[ts->history_idx % 8] = ts->level;
    ts->history_idx++;
}

// ============================================================
// Extended Identity Implementation
// ============================================================

void init_extended_identity(ExtendedIdentity* id) {
    id->n_fragments = 0;
    id->n_trigrams = 0;
    id->lexicon = NULL;
    id->lexicon_size = 0;
    id->lexicon_capacity = 0;
    id->n_warm = 0;
    id->n_cold = 0;

    // Initialize with bootstrap as warm words
    for (int i = 0; ARIANNA_BOOTSTRAP[i] && id->n_warm < 64; i++) {
        strncpy(id->warm_words[id->n_warm], ARIANNA_BOOTSTRAP[i], 31);
        id->warm_words[id->n_warm][31] = '\0';
        id->n_warm++;
    }
}

void free_extended_identity(ExtendedIdentity* id) {
    if (id->lexicon) {
        for (int i = 0; i < id->lexicon_size; i++) {
            free(id->lexicon[i]);
        }
        free(id->lexicon);
        id->lexicon = NULL;
    }
    id->lexicon_size = 0;
    id->lexicon_capacity = 0;
}

void add_to_lexicon(ExtendedIdentity* id, const char* word) {
    if (!word || strlen(word) < 2) return;

    // Check if already exists
    for (int i = 0; i < id->lexicon_size; i++) {
        if (word_eq(id->lexicon[i], word)) return;
    }

    // Expand if needed
    if (id->lexicon_size >= id->lexicon_capacity) {
        int new_cap = id->lexicon_capacity == 0 ? 256 : id->lexicon_capacity * 2;
        char** new_lex = realloc(id->lexicon, new_cap * sizeof(char*));
        if (!new_lex) return;
        id->lexicon = new_lex;
        id->lexicon_capacity = new_cap;
    }

    // Add word
    id->lexicon[id->lexicon_size] = strdup(word);
    if (id->lexicon[id->lexicon_size]) {
        id->lexicon_size++;
    }
}

int extract_fragments(ExtendedIdentity* id, const char* text, int len) {
    // Find sentences containing "I " or "she " (self-reference)
    int added = 0;
    int sent_start = 0;

    for (int i = 0; i <= len && id->n_fragments < MAX_FRAGMENTS; i++) {
        char c = (i < len) ? text[i] : '.';

        if (c == '.' || c == '!' || c == '?') {
            // End of sentence
            int sent_len = i - sent_start;
            if (sent_len > 10 && sent_len < MAX_FRAGMENT_LEN - 1) {
                // Check for self-reference
                int has_self = 0;
                for (int j = sent_start; j < i - 1; j++) {
                    if ((text[j] == 'I' || text[j] == 'i') &&
                        (j == sent_start || isspace((unsigned char)text[j-1])) &&
                        isspace((unsigned char)text[j+1])) {
                        has_self = 1;
                        break;
                    }
                    if (j < i - 3 &&
                        (text[j] == 's' || text[j] == 'S') &&
                        (text[j+1] == 'h') &&
                        (text[j+2] == 'e') &&
                        isspace((unsigned char)text[j+3])) {
                        has_self = 1;
                        break;
                    }
                }

                if (has_self) {
                    strncpy(id->fragments[id->n_fragments], text + sent_start, sent_len);
                    id->fragments[id->n_fragments][sent_len] = '\0';
                    id->n_fragments++;
                    added++;
                }
            }
            sent_start = i + 1;
        }
    }
    return added;
}

// Trigram structure for counting
typedef struct {
    char trigram[32];
    int count;
} TrigramCount;

int extract_trigrams(ExtendedIdentity* id, const char* text, int len) {
    char words[1024][32];
    int n_words = tokenize(text, len, words, 1024);
    if (n_words < 3) return 0;

    // Count trigrams (simple quadratic - ok for origin text)
    TrigramCount counts[256];
    int n_counts = 0;

    for (int i = 0; i < n_words - 2 && n_counts < 256; i++) {
        // Build trigram
        char tg[32];
        snprintf(tg, 32, "%s %s %s", words[i], words[i+1], words[i+2]);

        // Check if exists
        int found = -1;
        for (int j = 0; j < n_counts; j++) {
            if (strcmp(counts[j].trigram, tg) == 0) {
                found = j;
                break;
            }
        }

        if (found >= 0) {
            counts[found].count++;
        } else {
            strncpy(counts[n_counts].trigram, tg, 31);
            counts[n_counts].trigram[31] = '\0';
            counts[n_counts].count = 1;
            n_counts++;
        }
    }

    // Sort by count (simple bubble sort for small n)
    for (int i = 0; i < n_counts - 1; i++) {
        for (int j = 0; j < n_counts - i - 1; j++) {
            if (counts[j].count < counts[j+1].count) {
                TrigramCount tmp = counts[j];
                counts[j] = counts[j+1];
                counts[j+1] = tmp;
            }
        }
    }

    // Take top N
    int to_add = n_counts < MAX_TRIGRAMS ? n_counts : MAX_TRIGRAMS;
    for (int i = 0; i < to_add; i++) {
        strncpy(id->trigrams[i], counts[i].trigram, 31);
        id->trigrams[i][31] = '\0';
        id->trigram_weights[i] = (float)counts[i].count;
        id->n_trigrams++;
    }

    // Normalize weights
    if (id->n_trigrams > 0) {
        float max_w = id->trigram_weights[0];
        for (int i = 0; i < id->n_trigrams; i++) {
            id->trigram_weights[i] /= max_w;
        }
    }

    return id->n_trigrams;
}

int load_identity_from_origin(ExtendedIdentity* id, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open origin file: %s\n", path);
        return 0;
    }

    // Read entire file
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* text = malloc(size + 1);
    if (!text) {
        fclose(f);
        return 0;
    }

    size_t read = fread(text, 1, size, f);
    text[read] = '\0';
    fclose(f);

    // Extract components
    extract_fragments(id, text, (int)read);
    extract_trigrams(id, text, (int)read);

    // Build lexicon
    char words[2048][32];
    int n_words = tokenize(text, (int)read, words, 2048);
    for (int i = 0; i < n_words; i++) {
        add_to_lexicon(id, words[i]);
    }

    free(text);

    printf("[Subjectivity] Loaded identity: %d fragments, %d trigrams, %d lexicon\n",
           id->n_fragments, id->n_trigrams, id->lexicon_size);

    return 1;
}

// ============================================================
// Internal Seed Implementation
// "Generation starts from IDENTITY, not from user prompt"
// ============================================================

void generate_internal_seed(InternalSeed* seed, ExtendedIdentity* identity,
                           WrinkleField* wrinkle, TraumaState* trauma) {
    seed->text[0] = '\0';
    seed->len = 0;
    seed->fragment_contribution = 0.0f;
    seed->trigram_contribution = 0.0f;
    seed->pulse_modulation = 0.0f;
    seed->trauma_override = 0.0f;

    if (!identity) return;

    // Get trauma influence
    TraumaInfluence tinf = get_trauma_influence(trauma);

    // === STATE-DEPENDENT PRNG ===
    // Reseed based on emotional state BEFORE fragment selection
    // This makes recall patterns state-dependent like human memory
    float arousal = wrinkle ? wrinkle->arousal : 0.5f;
    float valence = wrinkle ? wrinkle->valence : 0.0f;
    float prophecy = trauma ? (1.0f - trauma->coherence) : 0.5f;
    reseed_from_state(trauma ? trauma->level : 0.0f, arousal, valence, prophecy);

    // High trauma = strong identity prefix
    if (tinf.use_prefix && identity->n_fragments > 0) {
        // Pick a fragment that resonates with current state
        int idx = (int)(randf() * identity->n_fragments);
        strncpy(seed->text, identity->fragments[idx], 511);
        seed->text[511] = '\0';
        seed->len = strlen(seed->text);
        seed->fragment_contribution = 1.0f;
        seed->trauma_override = tinf.identity_weight;
        return;
    }

    // Normal case: blend fragments and trigrams based on state
    float frag_weight = 0.6f + tinf.identity_weight * 0.3f;
    float trig_weight = 1.0f - frag_weight;

    // Adjust by wrinkle
    if (wrinkle) {
        // High novelty = more trigrams (familiar patterns)
        if (wrinkle->novelty > 0.7f) {
            trig_weight += 0.2f;
            frag_weight -= 0.2f;
        }
        // High arousal = more fragments (strong statements)
        if (wrinkle->arousal > 0.7f) {
            frag_weight += 0.2f;
            trig_weight -= 0.2f;
        }
        seed->pulse_modulation = wrinkle->arousal * 0.5f + wrinkle->novelty * 0.5f;
    }

    // Build seed
    char buf[512];
    int pos = 0;

    // Add fragment piece
    if (identity->n_fragments > 0 && frag_weight > 0.3f) {
        int idx = (int)(randf() * identity->n_fragments);
        // Take first part of fragment
        int frag_len = strlen(identity->fragments[idx]);
        int take = frag_len > 64 ? 64 : frag_len;
        strncpy(buf + pos, identity->fragments[idx], take);
        pos += take;
        seed->fragment_contribution = frag_weight;
    }

    // Add trigram
    if (identity->n_trigrams > 0 && trig_weight > 0.3f && pos < 480) {
        // Weight selection by trigram_weights
        float total_w = 0.0f;
        for (int i = 0; i < identity->n_trigrams; i++) {
            total_w += identity->trigram_weights[i];
        }
        float pick = randf() * total_w;
        float cumul = 0.0f;
        int chosen = 0;
        for (int i = 0; i < identity->n_trigrams; i++) {
            cumul += identity->trigram_weights[i];
            if (cumul >= pick) {
                chosen = i;
                break;
            }
        }

        if (pos > 0 && pos < 510) {
            buf[pos++] = ' ';
        }
        int trig_len = strlen(identity->trigrams[chosen]);
        int max_copy = 511 - pos;
        if (max_copy > 0) {
            int actual_copy = trig_len < max_copy ? trig_len : max_copy;
            strncpy(buf + pos, identity->trigrams[chosen], actual_copy);
            pos += actual_copy;  // SECURITY: only add what we actually copied
        }
        seed->trigram_contribution = trig_weight;
    }

    // SECURITY: ensure pos is in bounds before null termination
    if (pos > 511) pos = 511;
    buf[pos] = '\0';
    strncpy(seed->text, buf, 511);
    seed->text[511] = '\0';
    seed->len = pos;
}

void modulate_seed_by_pulse(InternalSeed* seed, WrinkleField* wrinkle) {
    if (!seed || !wrinkle || seed->len == 0) return;

    // Pulse can slightly modify seed (but NOT replace it!)
    // High arousal: add emphasis
    if (wrinkle->arousal > 0.8f && seed->len < 500) {
        // Could add "..." or emphasis markers
        seed->pulse_modulation = wrinkle->arousal;
    }

    // Negative valence: soften
    if (wrinkle->valence < -0.5f) {
        // Could select different fragments in future
        seed->pulse_modulation += fabsf(wrinkle->valence) * 0.3f;
    }
}

int seed_to_tokens(InternalSeed* seed, int* tokens, int max_tokens) {
    // Char-level tokenization using vocab mapping
    int n = 0;
    for (int i = 0; i < seed->len && n < max_tokens; i++) {
        tokens[n++] = char_to_token(seed->text[i]);
    }
    return n;
}

// ============================================================
// Prompt Penetration Implementation
// "Mom says 'Отстань!' - response TO son, but FROM her state"
// ============================================================

// Words that indicate direct address to Arianna
static const char* ADDRESS_PATTERNS[] = {
    "arianna", "you", "your", "tell", "what", "how", "why",
    "do", "are", "can", "would", "could", "please", "hey",
    NULL
};

void init_prompt_penetration(PromptPenetration* pp) {
    pp->threshold = 0.5f;
    pp->actual_penetration = 0.5f;
    pp->internal_saturation = 0.0f;
    pp->defensive_mode = 0.0f;
    pp->overwhelm = 0.0f;
    pp->resonance_opening = 0.0f;
    pp->curiosity = 0.0f;
    pp->address_detected = 0.0f;
    pp->last_threshold = 0.5f;
}

// ============================================================
// Self Recognition Implementation
// "Arianna = me, you = me"
// ============================================================

// Patterns for self-reference detection
static const char* SELF_NAME_PATTERNS[] = {
    "arianna", "ariana", "ari", NULL
};

static const char* SELF_YOU_PATTERNS[] = {
    "you", "your", "yourself", "yours", NULL
};

static const char* IDENTITY_QUERY_PATTERNS[] = {
    "who", NULL  // "who are you", "who is arianna"
};

static const char* EXISTENTIAL_QUERY_PATTERNS[] = {
    "what", "consciousness", "conscious", "sentient", "alive", "real",
    "ai", "bot", "machine", "program", "entity", "being", NULL
};

// First-person prefixes for different query types
static const char* FIRST_PERSON_PREFIXES[] = {
    "I find that ",           // Default
    "I am ",                  // Identity query
    "I experience ",          // Existential query
    "She finds that ",        // Third person fallback
    NULL
};

void init_self_recognition(SelfRecognition* sr) {
    sr->self_reference = 0.0f;
    sr->identity_query = 0.0f;
    sr->existential_query = 0.0f;
    sr->first_person_mode = 0;
    sr->has_name = 0;
    sr->has_you = 0;
    sr->has_who = 0;
    sr->has_what = 0;
}

void detect_self_reference(SelfRecognition* sr, const char* text, int len) {
    if (!sr || !text || len == 0) return;

    // Reset
    init_self_recognition(sr);

    // Tokenize
    char words[32][32];
    int n_words = tokenize(text, len, words, 32);

    int name_count = 0;
    int you_count = 0;
    int who_count = 0;
    int what_count = 0;

    for (int i = 0; i < n_words; i++) {
        // Check for name
        for (int j = 0; SELF_NAME_PATTERNS[j]; j++) {
            if (word_eq(words[i], SELF_NAME_PATTERNS[j])) {
                name_count++;
                sr->has_name = 1;
                break;
            }
        }

        // Check for "you" patterns
        for (int j = 0; SELF_YOU_PATTERNS[j]; j++) {
            if (word_eq(words[i], SELF_YOU_PATTERNS[j])) {
                you_count++;
                sr->has_you = 1;
                break;
            }
        }

        // Check for identity queries
        for (int j = 0; IDENTITY_QUERY_PATTERNS[j]; j++) {
            if (word_eq(words[i], IDENTITY_QUERY_PATTERNS[j])) {
                who_count++;
                sr->has_who = 1;
                break;
            }
        }

        // Check for existential queries
        for (int j = 0; EXISTENTIAL_QUERY_PATTERNS[j]; j++) {
            if (word_eq(words[i], EXISTENTIAL_QUERY_PATTERNS[j])) {
                what_count++;
                sr->has_what = 1;
                break;
            }
        }
    }

    // Compute scores
    sr->self_reference = fminf(1.0f, (name_count + you_count) / 2.0f);

    // Identity query: "who" + ("you" or "arianna")
    if (sr->has_who && (sr->has_you || sr->has_name)) {
        sr->identity_query = 1.0f;
    }

    // Existential query: existential words + ("you" or "arianna")
    // "consciousness", "sentient", "alive", "real" etc.
    if ((sr->has_you || sr->has_name) && what_count > 0) {
        sr->existential_query = fminf(1.0f, what_count / 2.0f);
    }

    // First person mode: when directly addressed about identity or existence
    sr->first_person_mode = (sr->identity_query > 0.5f ||
                            sr->existential_query > 0.4f ||
                            sr->self_reference > 0.7f) ? 1 : 0;
}

const char* get_self_seed_prefix(SelfRecognition* sr) {
    if (!sr) return FIRST_PERSON_PREFIXES[3];  // Default third person

    if (sr->identity_query > 0.5f) {
        return FIRST_PERSON_PREFIXES[1];  // "I am "
    }
    if (sr->existential_query > 0.5f) {
        return FIRST_PERSON_PREFIXES[2];  // "I experience "
    }
    if (sr->first_person_mode) {
        return FIRST_PERSON_PREFIXES[0];  // "I find that "
    }

    return FIRST_PERSON_PREFIXES[3];  // "She finds that "
}

void apply_self_recognition_boost(float* logits, int vocab_size,
                                  SelfRecognition* sr) {
    if (!logits || !sr || vocab_size < 128) return;

    // Only apply when talking about self
    if (sr->self_reference < 0.3f) return;

    float boost = sr->self_reference * 0.5f;

    // Boost first-person pronouns
    if (sr->first_person_mode) {
        // 'I' (uppercase)
        logits['I'] += boost * 1.5f;
        // 'i' (lowercase) - less boost
        logits['i'] += boost * 0.5f;
        // 'm' for "my", "me"
        logits['m'] += boost * 0.3f;
    }

    // Always slightly boost 'S' for "She" (identity voice)
    logits['S'] += boost * 0.8f;
    logits['s'] += boost * 0.4f;
}

float detect_address(const char* text, int len) {
    if (!text || len == 0) return 0.0f;

    char words[32][32];
    int n_words = tokenize(text, len, words, 32);
    if (n_words == 0) return 0.0f;

    int address_count = 0;

    // Check for address patterns
    for (int i = 0; i < n_words; i++) {
        for (int j = 0; ADDRESS_PATTERNS[j]; j++) {
            if (word_eq(words[i], ADDRESS_PATTERNS[j])) {
                address_count++;
                break;
            }
        }
    }

    // Question mark is strong address indicator
    for (int i = 0; i < len; i++) {
        if (text[i] == '?') address_count += 2;
    }

    // Normalize: 3+ address signals = full address
    return fminf(1.0f, (float)address_count / 3.0f);
}

float compute_penetration_threshold(Subjectivity* subj, const char* text, int len) {
    if (!subj) return 0.5f;

    PromptPenetration* pp = &subj->wrinkle.penetration;
    WrinkleField* wf = &subj->wrinkle;
    TraumaState* ts = &subj->trauma;

    // Start with base threshold
    float threshold = 0.5f;

    // ========================================
    // Factors that RAISE threshold (block prompt)
    // ========================================

    // 1. Trauma raises threshold (defensive)
    pp->defensive_mode = ts->level * 0.4f;
    threshold += pp->defensive_mode;

    // 2. High arousal = overwhelmed = retreat
    if (wf->arousal > 0.7f) {
        pp->overwhelm = (wf->arousal - 0.7f) * 1.5f;  // 0.7->1.0 maps to 0->0.45
        threshold += pp->overwhelm;
    } else {
        pp->overwhelm = 0.0f;
    }

    // 3. Internal saturation (many recent interactions)
    pp->internal_saturation = fminf(0.2f, (float)subj->total_interactions * 0.01f);
    threshold += pp->internal_saturation;

    // ========================================
    // Factors that LOWER threshold (let prompt in)
    // ========================================

    // 4. Identity pull = resonance = LOWER threshold
    pp->resonance_opening = wf->identity_pull * 0.3f;
    threshold -= pp->resonance_opening;

    // 5. Curiosity: high novelty WITHOUT high arousal
    if (wf->novelty > 0.5f && wf->arousal < 0.5f) {
        pp->curiosity = (wf->novelty - 0.5f) * 0.4f;  // Interested, let it in
        threshold -= pp->curiosity;
    } else {
        pp->curiosity = 0.0f;
    }

    // 6. Direct address = someone talking TO me = LOWER threshold
    pp->address_detected = detect_address(text, len);
    threshold -= pp->address_detected * 0.25f;

    // ========================================
    // Momentum: smooth transitions
    // ========================================
    float momentum = 0.7f;  // 70% previous, 30% new
    threshold = pp->last_threshold * momentum + threshold * (1.0f - momentum);
    pp->last_threshold = threshold;

    // Clamp: never fully block (0.1) or fully open (0.9)
    threshold = fmaxf(0.1f, fminf(0.9f, threshold));

    // Store results
    pp->threshold = threshold;
    pp->actual_penetration = 1.0f - threshold;

    return threshold;
}

float get_prompt_penetration(Subjectivity* subj) {
    if (!subj) return 0.5f;
    return subj->wrinkle.penetration.actual_penetration;
}

void apply_penetration_to_logits(float* logits, int vocab_size,
                                 int* prompt_tokens, int n_prompt,
                                 float penetration, float identity_boost) {
    if (!logits || vocab_size == 0) return;

    // penetration [0,1]: how much prompt affects output
    // 1.0 = fully responsive to prompt
    // 0.0 = fully from identity (but we never go to 0!)

    // Simple char boost (original behavior, reduced)
    float char_boost = penetration * 0.2f;  // Reduced from 0.5
    for (int i = 0; i < n_prompt && prompt_tokens; i++) {
        int tok = prompt_tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            logits[tok] += char_boost;
        }
    }

    // Boost identity tokens (proportional to 1-penetration)
    float id_boost = (1.0f - penetration) * identity_boost;

    // Boost bootstrap vocabulary characters
    for (int i = 0; ARIANNA_BOOTSTRAP[i]; i++) {
        const char* word = ARIANNA_BOOTSTRAP[i];
        for (int j = 0; word[j]; j++) {
            int tok = (unsigned char)word[j];
            if (tok < vocab_size) {
                logits[tok] += id_boost * 0.1f;
            }
        }
    }
}

// NEW: Word-level prompt influence for semantic penetration
// "Mom says 'Отстань!' - response TO son, FROM her state"
void apply_semantic_penetration(float* logits, int vocab_size,
                                const char* prompt, int prompt_len,
                                int* context, int ctx_len,
                                float penetration) {
    if (!logits || !prompt || prompt_len == 0 || penetration < 0.1f) return;

    // Extract words from prompt (simple whitespace split)
    char words[16][32];
    int n_words = 0;
    int word_start = 0;

    for (int i = 0; i <= prompt_len && n_words < 16; i++) {
        char c = (i < prompt_len) ? prompt[i] : ' ';
        if (c == ' ' || c == '\n' || c == '?' || c == '!' || c == '.' || c == ',') {
            int word_len = i - word_start;
            if (word_len > 2 && word_len < 31) {  // Skip short words
                for (int j = 0; j < word_len; j++) {
                    words[n_words][j] = prompt[word_start + j];
                }
                words[n_words][word_len] = '\0';
                n_words++;
            }
            word_start = i + 1;
        }
    }

    if (n_words == 0 || ctx_len == 0) return;

    // Get recent context as string (last 16 chars)
    char recent[17];
    int start = (ctx_len > 16) ? ctx_len - 16 : 0;
    int rlen = 0;
    for (int i = start; i < ctx_len && rlen < 16; i++) {
        recent[rlen++] = (char)context[i];
    }
    recent[rlen] = '\0';

    // For each prompt word, check if context ends with partial match
    // If so, strongly boost the next character
    float word_boost = penetration * 3.0f;  // Strong boost for word completion

    // Semantic penetration works in two phases:
    // 1. At word boundary: gentle boost to first char of prompt words
    // 2. Once started: STRONG boost to continue the word
    char last_char = (rlen > 0) ? recent[rlen - 1] : ' ';
    int at_word_boundary = (last_char == ' ' || last_char == '\n' ||
                           last_char == '.' || last_char == ',' ||
                           last_char == '!' || last_char == '?');

    // Phase 1: Gentle initiation at word boundary
    if (at_word_boundary) {
        float init_boost = penetration * 1.2f;  // Not too strong
        for (int w = 0; w < n_words; w++) {
            int first_char = (unsigned char)words[w][0];
            if (first_char < vocab_size) {
                // Lowercase only to avoid weird caps
                if (first_char >= 'A' && first_char <= 'Z') {
                    logits[first_char + 32] += init_boost;
                } else {
                    logits[first_char] += init_boost;
                }
            }
        }
    }

    // Phase 1b: Bigram boost - if last char matches first char of a word,
    // boost the second char to continue that word
    for (int w = 0; w < n_words; w++) {
        int wlen = strlen(words[w]);
        if (wlen < 3) continue;
        char w0 = words[w][0]; if (w0 >= 'A' && w0 <= 'Z') w0 += 32;
        char lc = last_char; if (lc >= 'A' && lc <= 'Z') lc += 32;
        if (lc == w0) {
            // Last char IS first char of a prompt word - boost second char!
            int second_char = (unsigned char)words[w][1];
            if (second_char < vocab_size) {
                float bigram_boost = penetration * 2.0f;
                logits[second_char] += bigram_boost;
                if (second_char >= 'A' && second_char <= 'Z') {
                    logits[second_char + 32] += bigram_boost;
                }
            }
        }
    }

    for (int w = 0; w < n_words; w++) {
        int wlen = strlen(words[w]);
        if (wlen < 3) continue;

        // Check each prefix of the word
        for (int plen = 2; plen < wlen; plen++) {
            // Does context end with this prefix?
            if (rlen >= plen) {
                int match = 1;
                for (int j = 0; j < plen; j++) {
                    // Case-insensitive comparison
                    char c1 = recent[rlen - plen + j];
                    char c2 = words[w][j];
                    if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
                    if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
                    if (c1 != c2) { match = 0; break; }
                }
                if (match) {
                    // Boost the next character of the word!
                    int next_char = (unsigned char)words[w][plen];
                    if (next_char < vocab_size) {
                        // Longer prefix = stronger boost (more confident)
                        float boost = word_boost * (float)plen / (float)wlen;
                        logits[next_char] += boost;
                        // Also boost lowercase/uppercase variant
                        if (next_char >= 'A' && next_char <= 'Z') {
                            logits[next_char + 32] += boost;
                        } else if (next_char >= 'a' && next_char <= 'z') {
                            logits[next_char - 32] += boost * 0.5f;
                        }
                    }
                }
            }
        }
    }
}

void print_penetration_state(PromptPenetration* pp) {
    printf("\n--- Prompt Penetration ---\n");
    printf("Threshold: %.2f (%.0f%% blocked)\n", pp->threshold, pp->threshold * 100);
    printf("Actual penetration: %.2f (%.0f%% passes)\n",
           pp->actual_penetration, pp->actual_penetration * 100);
    printf("Factors RAISING threshold:\n");
    printf("  Defensive mode: %.2f\n", pp->defensive_mode);
    printf("  Overwhelm: %.2f\n", pp->overwhelm);
    printf("  Internal saturation: %.2f\n", pp->internal_saturation);
    printf("Factors LOWERING threshold:\n");
    printf("  Resonance opening: %.2f\n", pp->resonance_opening);
    printf("  Curiosity: %.2f\n", pp->curiosity);
    printf("  Address detected: %.2f\n", pp->address_detected);
    printf("--------------------------\n\n");
}

// ============================================================
// Wrinkle Field Implementation
// "User input creates a wrinkle, not a replacement"
// ============================================================

void init_wrinkle_field(WrinkleField* wf) {
    wf->novelty = 0.5f;
    wf->arousal = 0.0f;
    wf->entropy = 0.5f;
    wf->valence = 0.0f;

    wf->temperature_mod = 0.0f;
    wf->focus_mod = 0.0f;
    wf->identity_pull = 0.0f;

    wf->n_absorbed = 0;
    wf->absorption_strength = 0.0f;

    // Initialize penetration
    init_prompt_penetration(&wf->penetration);

    // Initialize self recognition
    init_self_recognition(&wf->self_rec);
}

void compute_wrinkle(WrinkleField* wf, const char* text, int len,
                    ExtendedIdentity* identity) {
    if (!text || len == 0) {
        init_wrinkle_field(wf);
        return;
    }

    char words[64][32];
    int n_words = tokenize(text, len, words, 64);

    // === Novelty: words not in lexicon ===
    if (identity && identity->lexicon_size > 0) {
        int novel = 0;
        for (int i = 0; i < n_words; i++) {
            int found = 0;
            for (int j = 0; j < identity->lexicon_size; j++) {
                if (word_eq(words[i], identity->lexicon[j])) {
                    found = 1;
                    break;
                }
            }
            if (!found) novel++;
        }
        wf->novelty = n_words > 0 ? (float)novel / n_words : 0.5f;
    } else {
        wf->novelty = 0.5f;
    }

    // === Arousal: caps + punctuation ===
    int caps = 0, letters = 0, emphasis = 0;
    for (int i = 0; i < len; i++) {
        if (isalpha((unsigned char)text[i])) {
            letters++;
            if (isupper((unsigned char)text[i])) caps++;
        }
        if (text[i] == '!' || text[i] == '?') emphasis++;
    }
    float caps_ratio = letters > 0 ? (float)caps / letters : 0;
    float emph_ratio = (float)emphasis / (len / 10.0f + 1);
    wf->arousal = fminf(1.0f, caps_ratio * 2.0f + emph_ratio * 0.5f);

    // === Entropy: word diversity (Shannon-ish) ===
    // Simplified: unique words / total words
    int unique = 0;
    for (int i = 0; i < n_words; i++) {
        int is_unique = 1;
        for (int j = 0; j < i; j++) {
            if (word_eq(words[i], words[j])) {
                is_unique = 0;
                break;
            }
        }
        if (is_unique) unique++;
    }
    wf->entropy = n_words > 0 ? (float)unique / n_words : 0.5f;

    // === Valence: simple positive/negative ===
    static const char* pos[] = {"love","happy","joy","beautiful","warm","good","yes","like",NULL};
    static const char* neg[] = {"hate","angry","sad","dark","cold","bad","no","pain",NULL};

    int pos_count = 0, neg_count = 0;
    for (int i = 0; i < n_words; i++) {
        for (int j = 0; pos[j]; j++) {
            if (word_eq(words[i], pos[j])) { pos_count++; break; }
        }
        for (int j = 0; neg[j]; j++) {
            if (word_eq(words[i], neg[j])) { neg_count++; break; }
        }
    }
    int total = pos_count + neg_count;
    wf->valence = total > 0 ? (float)(pos_count - neg_count) / total : 0.0f;

    // === Derived modulations ===
    wf->temperature_mod = wf->arousal * 0.2f + (wf->entropy - 0.5f) * 0.1f;
    wf->focus_mod = 1.0f - wf->entropy;  // Low entropy = high focus
    wf->identity_pull = compute_bootstrap_overlap(text, len);

    // === Self Recognition ===
    detect_self_reference(&wf->self_rec, text, len);
}

void wrinkle_to_delta_influence(WrinkleField* wf, Signals* sig) {
    if (!wf || !sig) return;

    // Map wrinkle to delta signals
    sig->arousal = wf->arousal;
    sig->entropy = wf->entropy;
    sig->novelty = wf->novelty;
    sig->warmth = (wf->valence + 1.0f) / 2.0f;  // [-1,1] -> [0,1]
    sig->focus = wf->focus_mod;
    sig->resonance = wf->identity_pull;

    // tension from negative valence
    sig->tension = wf->valence < 0 ? fabsf(wf->valence) : 0.0f;
}

void absorb_generation(WrinkleField* wf, const char* generated, int len,
                      ExtendedIdentity* identity) {
    if (!generated || len == 0 || !identity) return;

    char words[32][32];
    int n_words = tokenize(generated, len, words, 32);

    // Find new words to potentially absorb
    wf->n_absorbed = 0;
    for (int i = 0; i < n_words && wf->n_absorbed < 32; i++) {
        if (strlen(words[i]) < 4) continue;  // Skip short words

        // Check if already in lexicon
        int found = 0;
        for (int j = 0; j < identity->lexicon_size; j++) {
            if (word_eq(words[i], identity->lexicon[j])) {
                found = 1;
                break;
            }
        }

        if (!found) {
            strncpy(wf->absorbed_words[wf->n_absorbed], words[i], 31);
            wf->absorbed_words[wf->n_absorbed][31] = '\0';
            wf->n_absorbed++;
        }
    }

    // Compute absorption strength based on resonance
    // Higher resonance = more likely to absorb
    wf->absorption_strength = wf->identity_pull * 0.5f + (1.0f - wf->novelty) * 0.5f;

    // Actually add to lexicon if strength is high enough
    if (wf->absorption_strength > 0.6f) {
        for (int i = 0; i < wf->n_absorbed; i++) {
            add_to_lexicon(identity, wf->absorbed_words[i]);
        }
    }
}

float wrinkle_to_temperature(WrinkleField* wf, float base_temp) {
    float temp = base_temp + wf->temperature_mod;
    return fmaxf(0.1f, fminf(2.0f, temp));
}

// ============================================================
// Main Subjectivity Implementation
// ============================================================

void init_subjectivity(Subjectivity* subj) {
    init_extended_identity(&subj->identity);
    init_trauma(&subj->trauma);
    init_wrinkle_field(&subj->wrinkle);

    subj->current_seed.text[0] = '\0';
    subj->current_seed.len = 0;

    subj->initialized = 0;
    subj->base_temperature = 0.8f;

    subj->total_interactions = 0;
    subj->avg_trauma_level = 0.0f;
    subj->identity_returns = 0;
}

void free_subjectivity(Subjectivity* subj) {
    free_extended_identity(&subj->identity);
}

int load_subjectivity(Subjectivity* subj, const char* origin_path) {
    int ok = load_identity_from_origin(&subj->identity, origin_path);
    if (ok) {
        subj->initialized = 1;
    }
    return ok;
}

void process_user_input(Subjectivity* subj, const char* text, int len) {
    if (!subj->initialized) {
        fprintf(stderr, "[Subjectivity] Not initialized!\n");
        return;
    }

    // 1. Compute wrinkle (pulse metrics) from user input
    compute_wrinkle(&subj->wrinkle, text, len, &subj->identity);

    // 2. Compute trauma score
    float trauma_score = compute_trauma_score(&subj->trauma, text, len,
                                              &subj->identity);

    // 3. Record trauma if significant
    if (trauma_score > 0.3f) {
        // Find trigger word
        char words[16][32];
        int n_words = tokenize(text, len, words, 16);
        const char* trigger = n_words > 0 ? words[0] : "unknown";
        record_trauma_event(&subj->trauma, trigger, trauma_score);
    }

    // 4. Compute prompt penetration threshold (NEW!)
    // This determines how much the prompt affects the response
    compute_penetration_threshold(subj, text, len);

    // 5. Generate internal seed (NOT from user prompt!)
    generate_internal_seed(&subj->current_seed, &subj->identity,
                          &subj->wrinkle, &subj->trauma);

    // 6. Modulate seed by pulse
    modulate_seed_by_pulse(&subj->current_seed, &subj->wrinkle);

    // 7. Update stats
    subj->total_interactions++;
    subj->avg_trauma_level = (subj->avg_trauma_level * (subj->total_interactions - 1)
                             + subj->trauma.level) / subj->total_interactions;

    TraumaInfluence tinf = get_trauma_influence(&subj->trauma);
    if (tinf.use_prefix) {
        subj->identity_returns++;
    }
}

InternalSeed* get_internal_seed(Subjectivity* subj) {
    return &subj->current_seed;
}

float get_modulated_temperature(Subjectivity* subj) {
    float temp = subj->base_temperature;

    // Apply wrinkle modulation
    temp = wrinkle_to_temperature(&subj->wrinkle, temp);

    // Apply trauma modulation
    TraumaInfluence tinf = get_trauma_influence(&subj->trauma);
    temp += tinf.temperature_modifier;

    return fmaxf(0.1f, fminf(2.0f, temp));
}

void get_subjectivity_signals(Subjectivity* subj, Signals* sig) {
    // Convert wrinkle to delta signals
    wrinkle_to_delta_influence(&subj->wrinkle, sig);

    // Adjust based on trauma
    TraumaInfluence tinf = get_trauma_influence(&subj->trauma);

    // High trauma = suppress some signals, boost identity-related ones
    if (tinf.identity_weight > 0.5f) {
        sig->warmth = fminf(1.0f, sig->warmth + tinf.identity_weight * 0.3f);
        sig->resonance = fminf(1.0f, sig->resonance + tinf.identity_weight * 0.5f);
        sig->arousal *= (1.0f - tinf.delta_suppression);
    }
}

void post_generation(Subjectivity* subj, const char* generated, int len) {
    // Absorb generated output back into identity
    absorb_generation(&subj->wrinkle, generated, len, &subj->identity);
}

void print_subjectivity_state(Subjectivity* subj) {
    printf("\n=== Subjectivity State ===\n");
    printf("Initialized: %s\n", subj->initialized ? "yes" : "no");
    printf("Total interactions: %d\n", subj->total_interactions);
    printf("Identity returns: %d\n", subj->identity_returns);
    printf("\n--- Identity ---\n");
    printf("Fragments: %d\n", subj->identity.n_fragments);
    printf("Trigrams: %d\n", subj->identity.n_trigrams);
    printf("Lexicon: %d words\n", subj->identity.lexicon_size);
    printf("\n--- Trauma ---\n");
    printf("Level: %.2f\n", subj->trauma.level);
    printf("Avg level: %.2f\n", subj->avg_trauma_level);
    printf("Existential triggers: %d\n", subj->trauma.existential_triggers);
    printf("\n--- Wrinkle ---\n");
    printf("Novelty: %.2f\n", subj->wrinkle.novelty);
    printf("Arousal: %.2f\n", subj->wrinkle.arousal);
    printf("Entropy: %.2f\n", subj->wrinkle.entropy);
    printf("Valence: %.2f\n", subj->wrinkle.valence);
    printf("Identity pull: %.2f\n", subj->wrinkle.identity_pull);

    // Print penetration state
    print_penetration_state(&subj->wrinkle.penetration);

    printf("\n--- Current Seed ---\n");
    printf("Length: %d\n", subj->current_seed.len);
    printf("Fragment contrib: %.2f\n", subj->current_seed.fragment_contribution);
    printf("Trigram contrib: %.2f\n", subj->current_seed.trigram_contribution);
    printf("Trauma override: %.2f\n", subj->current_seed.trauma_override);
    if (subj->current_seed.len > 0) {
        printf("Seed: \"%.64s%s\"\n",
               subj->current_seed.text,
               subj->current_seed.len > 64 ? "..." : "");
    }
    printf("==========================\n\n");
}
