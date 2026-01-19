/*
 * julia_bridge.h — C bridge to Julia emotional engine
 * ═══════════════════════════════════════════════════════════════════════════════
 * הגשר בין סי לג'וליה
 * The bridge from C to Julia
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#ifndef JULIA_BRIDGE_H
#define JULIA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * TYPES
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Full emotional analysis result from Julia
 *
 * Primary emotions (Plutchik's wheel + Arianna extensions):
 *   joy, trust, fear, surprise, sadness, disgust, anger, anticipation
 *   + resonance, presence, longing, wonder
 *
 * Tertiary nuances (what discrete systems miss):
 *   bittersweetness, nostalgia, serenity, melancholy, tenderness,
 *   vulnerability, wistfulness, euphoria, desolation, reverence,
 *   compassion, ecstasy
 */
typedef struct {
    /* Primary emotions (12D) */
    float joy;
    float trust;
    float fear;
    float surprise;
    float sadness;
    float disgust;
    float anger;
    float anticipation;
    float resonance;
    float presence;
    float longing;
    float wonder;

    /* Tertiary nuances */
    float bittersweetness;
    float nostalgia;
    float serenity;
    float melancholy;
    float tenderness;
    float vulnerability;
    float wistfulness;
    float euphoria;
    float desolation;
    float reverence;
    float compassion;
    float ecstasy;
} JuliaEmotionalResult;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Initialize Julia bridge
 * Spawns Julia subprocess with emotional engine
 *
 * Returns: 1 on success, 0 if Julia not available
 *
 * Note: Called automatically on first use, but can be called
 *       explicitly for eager initialization
 */
int julia_init(void);

/*
 * Shutdown Julia bridge
 * Terminates Julia subprocess
 */
void julia_shutdown(void);

/*
 * Check if Julia is available
 *
 * Returns: 1 if Julia is initialized and running, 0 otherwise
 */
int julia_is_available(void);

/* ═══════════════════════════════════════════════════════════════════════════════
 * TEXT ANALYSIS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Analyze text and compute emotional state
 *
 * text:   Input text to analyze
 * result: Output emotional result (filled on success)
 *
 * Returns: 1 on success, 0 on failure
 *
 * Example:
 *   JuliaEmotionalResult result;
 *   if (julia_analyze_text("I feel a bittersweet longing", &result)) {
 *       printf("Nostalgia: %.3f\n", result.nostalgia);
 *       printf("Longing: %.3f\n", result.longing);
 *   }
 */
int julia_analyze_text(const char* text, JuliaEmotionalResult* result);

/* ═══════════════════════════════════════════════════════════════════════════════
 * GRADIENT COMPUTATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Compute emotional gradient between two states
 *
 * from:      12D emotional state (start)
 * to:        12D emotional state (end)
 * direction: Output 12D direction vector (unit)
 * magnitude: Output magnitude of change
 *
 * Returns: 1 on success, 0 on failure
 */
int julia_compute_gradient(const float* from, const float* to,
                           float* direction, float* magnitude);

/* ═══════════════════════════════════════════════════════════════════════════════
 * ODE DYNAMICS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Step emotional state forward in time using ODE
 *
 * state:     Current 12D emotional state
 * input:     12D external input (stimulus)
 * dt:        Time step
 * new_state: Output 12D new emotional state
 *
 * Returns: 1 on success, 0 on failure
 *
 * Note: Uses Arianna's default emotional dynamics parameters
 */
int julia_step_emotion(const float* state, const float* input, float dt,
                       float* new_state);

/* ═══════════════════════════════════════════════════════════════════════════════
 * RESONANCE
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Compute resonance between internal and external emotional states
 *
 * internal: 12D internal emotional state
 * external: 12D external emotional state (from input)
 *
 * Returns: Resonance value in [-1, 1]
 *          1 = perfect alignment (in phase)
 *         -1 = opposite (anti-phase)
 *          0 = no resonance
 */
float julia_compute_resonance(const float* internal, const float* external);

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEBUG
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Print emotional result to stdout
 */
void julia_print_result(const JuliaEmotionalResult* r);

#ifdef __cplusplus
}
#endif

#endif /* JULIA_BRIDGE_H */
