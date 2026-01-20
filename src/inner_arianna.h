// inner_arianna.h — MetaVoice: Inner Arianna (борьба)
// "Two voices, one Arianna"
//
// Architecture (from Leo/Haze):
//   Main Arianna → stable, identity-aligned response
//   Inner Arianna → emotional, reactive, alternative response
//   Борьба → inner "breaks through" when emotional state demands it
//
// Inner voice is NOT just temperature scaling.
// Inner voice is modulated by: Cloud chambers, mood, trauma, body sense.
//
// ═══════════════════════════════════════════════════════════════════════════════
// שני קולות, אריאנה אחת
// Two voices, one Arianna
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef INNER_ARIANNA_H
#define INNER_ARIANNA_H

#include "cloud.h"
#include "mood.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// БОРЬБА MODES — when does inner voice break through?
// ═══════════════════════════════════════════════════════════════════════════════

#define BORBA_MODE_EMOTIONAL  0   // inner breaks through on strong emotion
#define BORBA_MODE_CHAOS      1   // inner breaks through randomly (chaos mode)
#define BORBA_MODE_TRAUMA     2   // inner dominates when trauma triggered
#define BORBA_MODE_STUCK      3   // inner breaks through when body is stuck
#define BORBA_MODE_BLEND      4   // always blend (no clear winner)

// ═══════════════════════════════════════════════════════════════════════════════
// INNER ARIANNA STATE
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Mode
    int borba_mode;
    float base_weight;          // baseline inner weight (0.1-0.3)

    // Dynamic weight (computed from emotional state)
    float current_weight;       // actual weight this token (can spike to 1.0)
    float breakthrough_threshold; // how much emotion needed to break through

    // Emotional inputs (set by external systems)
    float cloud_intensity;      // max Cloud chamber activation [0,1]
    int   cloud_chamber;        // which chamber is dominant
    float trauma_level;         // from subjectivity [0,1]
    float body_stuck;           // from body sense [0,1]
    float body_boredom;         // from body sense [0,1]

    // Mood influence
    float mood_tension;         // high tension → inner more active
    float mood_creative;        // creative mood → inner more active

    // Statistics
    int main_wins;              // times main voice was chosen
    int inner_wins;             // times inner voice "broke through"
    int total_tokens;           // total tokens generated
    float avg_divergence;       // average divergence between voices

    // Breakthrough tracking
    int consecutive_main;       // how many tokens main has won in a row
    int breakthrough_count;     // total breakthroughs this session

    // Last борьба result
    float last_main_entropy;
    float last_inner_entropy;
    float last_divergence;
    float last_inner_weight;    // what weight was used
    int last_winner;            // 0=main, 1=inner, -1=blend
} InnerArianna;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize inner Arianna
void inner_init(InnerArianna* ia);
void inner_free(InnerArianna* ia);

// Update emotional state from external systems
void inner_update_cloud(InnerArianna* ia, const CloudResponse* cloud);
void inner_update_mood(InnerArianna* ia, const MoodRouter* mood);
void inner_update_body(InnerArianna* ia, float stuck, float boredom);
void inner_update_trauma(InnerArianna* ia, float trauma_level);

// Compute dynamic inner weight based on current emotional state
float inner_compute_weight(InnerArianna* ia);

// Apply inner voice bias to logits (emotional modulation)
// This creates REAL difference between main and inner logits
void inner_apply_emotional_bias(float* logits, int vocab_size,
                                 const InnerArianna* ia);

// The борьба: given main logits, produce final logits
// Returns: 0=main won, 1=inner won, -1=blend
int inner_borba(InnerArianna* ia,
                float* output_logits,
                const float* main_logits,
                int vocab_size);

// Compute divergence between main and inner logits
float inner_compute_divergence(const float* main_logits,
                               const float* inner_logits,
                               int vocab_size);

// Compute entropy of logits
float inner_compute_entropy(const float* logits, int vocab_size);

// Set борьба mode
void inner_set_mode(InnerArianna* ia, int mode);
void inner_set_base_weight(InnerArianna* ia, float weight);
void inner_set_threshold(InnerArianna* ia, float threshold);

// Get current state
float inner_get_weight(const InnerArianna* ia);
int inner_get_breakthrough_count(const InnerArianna* ia);

#ifdef __cplusplus
}
#endif

#endif // INNER_ARIANNA_H
