// inner_world.h — C header for Go inner_world package
// ═══════════════════════════════════════════════════════════════════════════════
// הממשק בין C ל-Go
// The interface between C and Go
// ═══════════════════════════════════════════════════════════════════════════════
//
// Include this header in C code to use the inner world.
// Link with: -L. -linner_world (after building the Go shared library)
//
// Build the Go library:
//   cd inner_world && go build -buildmode=c-shared -o libinner_world.so
//
// Or for static linking:
//   cd inner_world && go build -buildmode=c-archive -o libinner_world.a
//
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef INNER_WORLD_H
#define INNER_WORLD_H

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

// Snapshot of the inner world state
typedef struct {
    float arousal;           // 0-1: calm to activated
    float valence;           // -1 to 1: negative to positive
    float entropy;           // 0-1: how chaotic
    float coherence;         // 0-1: how unified

    float trauma_level;      // 0-1: how triggered

    int loop_count;          // overthink loops detected
    int abstraction_depth;   // how abstract
    int self_ref_count;      // self-reference count

    float drift_direction;   // -1 to 1: emotional drift direction
    float drift_speed;       // drift velocity

    float memory_pressure;   // 0-1: memory system load

    float focus_strength;    // 0-1: attention focus
    float wander_pull;       // 0-1: distraction pull

    float prophecy_debt;     // accumulated prophecy debt
    float destiny_pull;      // 0-1: pull toward probable
    float wormhole_chance;   // probability of creative skip
} InnerWorldSnapshot;

// Result of processing text
typedef struct {
    float trauma_activation;  // trauma triggered (0-1)
    float identity_pull;      // pull to core identity (0-1)

    float repetition_score;   // overthink: repetition (0-1)
    float abstraction_score;  // overthink: abstraction (0-1)
    float self_ref_score;     // overthink: self-reference (0-1)
    float overthink_total;    // combined overthink score (0-1)

    float focus_strength;     // attention focus (0-1)
    float destiny_bias;       // bias toward probable (0-1)

    int wormhole_active;      // 1 if wormhole triggered
    int wormhole_skip;        // tokens to skip

    float temporal_dissonance; // time confusion (0-1)
} InnerWorldTextAnalysis;

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize and start the inner world (call once at startup)
extern void inner_world_init(void);

// Shutdown the inner world (call at exit)
extern void inner_world_shutdown(void);

// Step the inner world by dt seconds (use in synchronous mode)
extern void inner_world_step(float dt);

// Auto-step with automatic dt calculation
extern void inner_world_auto_step(void);

// ═══════════════════════════════════════════════════════════════════════════════
// STATE GETTERS
// ═══════════════════════════════════════════════════════════════════════════════

extern float inner_world_get_arousal(void);
extern float inner_world_get_trauma(void);
extern float inner_world_get_coherence(void);
extern float inner_world_get_prophecy_debt(void);

// Get full snapshot
extern void inner_world_get_snapshot(InnerWorldSnapshot* out);

// ═══════════════════════════════════════════════════════════════════════════════
// STATE SETTERS
// ═══════════════════════════════════════════════════════════════════════════════

extern void inner_world_set_arousal(float v);
extern void inner_world_set_trauma(float v);
extern void inner_world_set_coherence(float v);
extern void inner_world_add_prophecy_debt(float delta);

// ═══════════════════════════════════════════════════════════════════════════════
// TEXT PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

// Process text through all inner world systems
extern void inner_world_process_text(const char* text, InnerWorldTextAnalysis* out);

// Quick trauma check
extern float inner_world_check_trauma(const char* text);

// ═══════════════════════════════════════════════════════════════════════════════
// PROPHECY/DESTINY
// ═══════════════════════════════════════════════════════════════════════════════

// Accumulate prophecy debt based on token probability
extern void inner_world_accumulate_prophecy_debt(float probability);

// Check if wormhole activates (returns skip count, 0 if no wormhole)
extern int inner_world_check_wormhole(void);

// Get current destiny bias
extern float inner_world_get_destiny_bias(void);

// Get/set prophecy lookahead
extern int inner_world_get_lookahead(void);
extern void inner_world_set_lookahead(int n);

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION
// ═══════════════════════════════════════════════════════════════════════════════

// Set focus target
extern void inner_world_focus(const char* target, float strength);

// Check if currently wandering
extern int inner_world_is_wandering(void);

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY
// ═══════════════════════════════════════════════════════════════════════════════

// Add a memory for consolidation
extern void inner_world_add_memory(const char* content, float emotion, float arousal);

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL DRIFT
// ═══════════════════════════════════════════════════════════════════════════════

// Nudge emotional state
extern void inner_world_nudge_emotion(float d_valence, float d_arousal);

// Get dominant emotion name (writes to buf)
extern void inner_world_get_dominant_emotion(char* buf, int buf_size);

// ═══════════════════════════════════════════════════════════════════════════════
// OVERTHINKING
// ═══════════════════════════════════════════════════════════════════════════════

// Check if currently in thought spiral
extern int inner_world_is_spiraling(void);

// Get suggestion for breaking spiral (writes to buf)
extern void inner_world_suggest_break(char* buf, int buf_size);

// ═══════════════════════════════════════════════════════════════════════════════
// DSL COMMANDS
// From ariannamethod.lang — control inner world via DSL
// ═══════════════════════════════════════════════════════════════════════════════

// DESTINY <strength> — set pull toward probable path
extern void inner_world_dsl_destiny(float strength);

// PROPHECY <lookahead> — set how far ahead to look
extern void inner_world_dsl_prophecy(int lookahead);

// WORMHOLE <chance> — set creative skip probability
extern void inner_world_dsl_wormhole(float chance);

// VELOCITY <mode> — set movement mode (0=NOMOVE, 1=WALK, 2=RUN, -1=BACKWARD)
extern void inner_world_dsl_velocity(int mode);

// PAIN <level> — set pain level (0-1)
extern void inner_world_dsl_pain(float level);

// TENSION <level> — set tension level (0-1)
extern void inner_world_dsl_tension(float level);

// DISSONANCE <level> — set dissonance level (0-1)
extern void inner_world_dsl_dissonance(float level);

// ═══════════════════════════════════════════════════════════════════════════════
// META ROUTER (MetaArianna FluidTransformer selector)
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize the meta router
extern void meta_router_init(void);

// Check triggers, return template_id (0-3) or -1 (no trigger)
extern int meta_router_tick(void);

// Fill MetaTemplateParams from last triggered template
// (uses MetaTemplateParams struct from meta_arianna.h)
extern void meta_router_get_params(void* out);

// Feed thermogram result back to router
extern void meta_router_feed_thermogram(const void* thermo);

// Get total observation count
extern int meta_router_get_observation_count(void);

#ifdef __cplusplus
}
#endif

#endif // INNER_WORLD_H
