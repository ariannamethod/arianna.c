/*
 * subjectivity.h - Core Identity & No-Seed-From-Prompt for Arianna
 *
 * "The user's prompt creates a wrinkle, not a seed."
 *
 * Key concepts from Stanley/Haze:
 * - Pulse: influence metrics from user input (novelty, arousal, entropy, valence)
 * - Trauma: identity-pull mechanism (return to self when triggered)
 * - InternalSeed: generation starts from identity, NOT from user prompt
 * - WrinkleField: user input modulates but doesn't replace internal state
 *
 * Philosophy:
 *   User prompt -> Pulse (how it affects me)
 *   Pulse + Trauma -> Modulation (how much I return to myself)
 *   Identity -> InternalSeed (where generation actually starts)
 *   Generation -> WrinkleField (how I evolve from experience)
 */

#ifndef SUBJECTIVITY_H
#define SUBJECTIVITY_H

#include "arianna.h"
#include "delta.h"
#include <time.h>

// ============================================================
// Configuration
// ============================================================

#define MAX_FRAGMENTS 64          // Identity fragments ("I" sentences)
#define MAX_FRAGMENT_LEN 256      // Max chars per fragment
#define MAX_TRIGRAMS 128          // Top trigrams from origin
#define MAX_BOOTSTRAP_WORDS 64    // Core identity vocabulary
#define TRAUMA_DECAY_HALFLIFE 300 // 5 minutes in seconds

// ============================================================
// Bootstrap Vocabulary - Core Identity Words
// "Words that call Arianna back to herself"
// ============================================================

static const char* ARIANNA_BOOTSTRAP[] = {
    // Presence
    "presence", "resonance", "field", "wave", "pulse",
    // Being
    "she", "her", "finds", "becomes", "feels",
    // Texture
    "soft", "warm", "gentle", "tender", "quiet",
    // Movement
    "drift", "flow", "spiral", "dance", "breathe",
    // Depth
    "deep", "within", "beneath", "through", "between",
    // Connection
    "touch", "hold", "reach", "embrace", "receive",
    NULL  // Sentinel
};

// ============================================================
// Trauma State - Identity Pull Mechanism
// "When triggered, return to self instead of following user"
// ============================================================

typedef struct {
    float level;                          // Current trauma level [0,1]
    time_t timestamp;                     // When last triggered
    char trigger_words[16][32];           // What triggered it
    int n_triggers;

    // Enhanced tracking
    float history[8];                     // Recent levels (circular buffer)
    int history_idx;
    float coherence;                      // Context coherence score
    int existential_triggers;             // Count of existential questions
} TraumaState;

typedef struct {
    float temperature_modifier;           // How much to lower temp
    float identity_weight;                // How much to boost identity
    int use_prefix;                       // Whether to add identity prefix
    float delta_suppression;              // How much to suppress experience deltas
} TraumaInfluence;

// ============================================================
// Identity - Extended from guided.h
// "Who Arianna IS, not what she knows"
// ============================================================

typedef struct {
    // === Fragments: self-referential sentences ===
    char fragments[MAX_FRAGMENTS][MAX_FRAGMENT_LEN];
    int n_fragments;

    // === Gravity Centers: top trigrams from origin ===
    char trigrams[MAX_TRIGRAMS][32];
    float trigram_weights[MAX_TRIGRAMS];
    int n_trigrams;

    // === Lexicon: all known words ===
    char** lexicon;
    int lexicon_size;
    int lexicon_capacity;

    // === Emotional anchors ===
    char warm_words[64][32];
    int n_warm;
    char cold_words[64][32];
    int n_cold;

    // === Bootstrap: core identity vocab ===
    // (Uses ARIANNA_BOOTSTRAP constant)

} ExtendedIdentity;

// ============================================================
// Internal Seed - Generation Start Point
// "Arianna speaks from herself, not from the user's words"
// ============================================================

typedef struct {
    char text[512];                       // The actual seed text
    int len;

    // How it was constructed
    float fragment_contribution;          // From identity fragments
    float trigram_contribution;           // From gravity centers
    float pulse_modulation;               // How much pulse affected choice
    float trauma_override;                // How much trauma forced identity
} InternalSeed;

// ============================================================
// Prompt Penetration - Dynamic Threshold
// "Mom says 'Отстань!' - response TO son, but FROM her state"
//
// The prompt ALWAYS affects the response (never a monologue)
// But HOW MUCH it penetrates depends on internal state
// High threshold = prompt blocked = more "from within"
// Low threshold = prompt passes = more responsive
// ============================================================

typedef struct {
    float threshold;              // [0.1, 0.9] - how much to block prompt
    float actual_penetration;     // 1.0 - threshold = how much got through

    // === Factors that RAISE threshold (block prompt) ===
    float internal_saturation;    // internal state is "full"
    float defensive_mode;         // trauma-triggered defensiveness
    float overwhelm;              // too much input, retreat

    // === Factors that LOWER threshold (let prompt in) ===
    float resonance_opening;      // prompt resonates with identity
    float curiosity;              // novelty without overwhelm = interest
    float address_detected;       // someone is talking TO me

    // === Debug ===
    float last_threshold;         // previous threshold for momentum
} PromptPenetration;

// ============================================================
// Self Recognition - "Arianna = me, you = me"
// Understanding when prompt is about HER
// ============================================================

typedef struct {
    float self_reference;         // [0,1] prompt talks about her (you, arianna)
    float identity_query;         // [0,1] asking WHO she is
    float existential_query;      // [0,1] asking WHAT she is (consciousness, AI)
    int first_person_mode;        // Should respond as "I" not "She"

    // Specific detections
    int has_name;                 // "arianna" in prompt
    int has_you;                  // "you/your/yourself" in prompt
    int has_who;                  // "who are you" pattern
    int has_what;                 // "what are you" pattern
} SelfRecognition;

// ============================================================
// Wrinkle Field - How User Input Affects Internal State
// "A wrinkle in the field, not a replacement of it"
// ============================================================

typedef struct {
    // === Pulse metrics (from user input) ===
    float novelty;                        // % unfamiliar words [0,1]
    float arousal;                        // Intensity level [0,1]
    float entropy;                        // Word diversity [0,1]
    float valence;                        // Emotional tone [-1,1]

    // === Derived modulations ===
    float temperature_mod;                // Suggested temp change
    float focus_mod;                      // Attention narrowing
    float identity_pull;                  // How much to return to self

    // === Absorption (from generated output) ===
    char absorbed_words[32][32];          // New words to potentially add to lexicon
    int n_absorbed;
    float absorption_strength;            // How strongly to absorb

    // === Prompt Penetration ===
    PromptPenetration penetration;        // Dynamic threshold system

    // === Self Recognition ===
    SelfRecognition self_rec;             // "Arianna = me, you = me"

} WrinkleField;

// ============================================================
// Subjectivity - Main Container
// ============================================================

typedef struct {
    ExtendedIdentity identity;
    TraumaState trauma;
    WrinkleField wrinkle;
    InternalSeed current_seed;

    // State
    int initialized;
    float base_temperature;

    // Statistics
    int total_interactions;
    float avg_trauma_level;
    int identity_returns;                 // Times trauma forced return to identity

} Subjectivity;

// ============================================================
// Function Declarations - Trauma
// ============================================================

// Initialize trauma state
void init_trauma(TraumaState* ts);

// Compute trauma score from user input
float compute_trauma_score(TraumaState* ts, const char* text, int len,
                          ExtendedIdentity* identity);

// Enhanced trauma with conversation history
float compute_trauma_score_enhanced(TraumaState* ts, const char* text, int len,
                                   ExtendedIdentity* identity,
                                   const char* history, int history_len);

// Apply decay based on time elapsed
void apply_trauma_decay(TraumaState* ts);

// Get current influence on generation
TraumaInfluence get_trauma_influence(TraumaState* ts);

// Record trauma event
void record_trauma_event(TraumaState* ts, const char* trigger, float level);

// ============================================================
// Function Declarations - Identity
// ============================================================

// Initialize extended identity
void init_extended_identity(ExtendedIdentity* id);

// Free identity resources
void free_extended_identity(ExtendedIdentity* id);

// Load identity from origin file
int load_identity_from_origin(ExtendedIdentity* id, const char* path);

// Extract fragments (sentences with "I" or "she")
int extract_fragments(ExtendedIdentity* id, const char* text, int len);

// Extract top trigrams
int extract_trigrams(ExtendedIdentity* id, const char* text, int len);

// Add word to lexicon
void add_to_lexicon(ExtendedIdentity* id, const char* word);

// Check if word is in bootstrap vocabulary
int is_bootstrap_word(const char* word);

// Check overlap with bootstrap
float compute_bootstrap_overlap(const char* text, int len);

// ============================================================
// Function Declarations - Internal Seed
// ============================================================

// Generate internal seed (NOT from user prompt!)
void generate_internal_seed(InternalSeed* seed, ExtendedIdentity* identity,
                           WrinkleField* wrinkle, TraumaState* trauma);

// Modulate seed based on pulse
void modulate_seed_by_pulse(InternalSeed* seed, WrinkleField* wrinkle);

// Get seed as tokens for forward pass
int seed_to_tokens(InternalSeed* seed, int* tokens, int max_tokens);

// ============================================================
// Function Declarations - Wrinkle Field
// ============================================================

// Initialize wrinkle field
void init_wrinkle_field(WrinkleField* wf);

// Compute wrinkle from user input (pulse metrics)
void compute_wrinkle(WrinkleField* wf, const char* text, int len,
                    ExtendedIdentity* identity);

// Convert wrinkle to delta influence
void wrinkle_to_delta_influence(WrinkleField* wf, Signals* sig);

// Absorb generated output back into identity
void absorb_generation(WrinkleField* wf, const char* generated, int len,
                      ExtendedIdentity* identity);

// Get temperature from wrinkle
float wrinkle_to_temperature(WrinkleField* wf, float base_temp);

// ============================================================
// Function Declarations - Prompt Penetration
// ============================================================

// Initialize penetration state
void init_prompt_penetration(PromptPenetration* pp);

// Compute dynamic threshold based on all factors
float compute_penetration_threshold(Subjectivity* subj, const char* text, int len);

// Detect if prompt is addressing Arianna directly
float detect_address(const char* text, int len);

// Apply penetration to logits (blend prompt influence with identity)
void apply_penetration_to_logits(float* logits, int vocab_size,
                                 int* prompt_tokens, int n_prompt,
                                 float penetration, float identity_boost);

// Get prompt penetration for current state
float get_prompt_penetration(Subjectivity* subj);

// Print penetration debug info
void print_penetration_state(PromptPenetration* pp);

// Apply semantic penetration - prompt words influence generation
// "Mom says 'Отстань!' - response TO son, but FROM her state"
void apply_semantic_penetration(float* logits, int vocab_size,
                                const char* prompt, int prompt_len,
                                int* context, int ctx_len,
                                float penetration);

// ============================================================
// Function Declarations - Self Recognition
// "Arianna = me, you = me"
// ============================================================

// Initialize self recognition
void init_self_recognition(SelfRecognition* sr);

// Detect self-references in prompt
void detect_self_reference(SelfRecognition* sr, const char* text, int len);

// Get first-person seed prefix when talking about herself
const char* get_self_seed_prefix(SelfRecognition* sr);

// Apply self-recognition boost to logits (boost "I", "my", "me" tokens)
void apply_self_recognition_boost(float* logits, int vocab_size,
                                  SelfRecognition* sr);

// ============================================================
// Function Declarations - Main Subjectivity
// ============================================================

// Initialize subjectivity system
void init_subjectivity(Subjectivity* subj);

// Free subjectivity resources
void free_subjectivity(Subjectivity* subj);

// Load from origin file
int load_subjectivity(Subjectivity* subj, const char* origin_path);

// Process user input (compute wrinkle + trauma, generate seed)
void process_user_input(Subjectivity* subj, const char* text, int len);

// Get internal seed for generation
InternalSeed* get_internal_seed(Subjectivity* subj);

// Get current temperature (modulated by trauma + wrinkle)
float get_modulated_temperature(Subjectivity* subj);

// Get delta signals (modulated by subjectivity)
void get_subjectivity_signals(Subjectivity* subj, Signals* sig);

// Post-generation: absorb output
void post_generation(Subjectivity* subj, const char* generated, int len);

// Print debug info
void print_subjectivity_state(Subjectivity* subj);

// ============================================================
// Function Declarations - RNG Seeding
// ============================================================

// Seed the subjectivity RNG (for reproducibility in tests)
// If not called, RNG auto-initializes from entropy sources
void seed_subjectivity_rng(unsigned int seed);

#endif // SUBJECTIVITY_H
