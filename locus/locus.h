// locus.h — Locus Coeruleus: Field Geometry Resonance Detector
// ═══════════════════════════════════════════════════════════════════════════════
// The "blue spot" in the brainstem. Releases norepinephrine when something
// important happens. Detects when field geometry demands SARTRE to speak.
// Stack-based. No syntax. Pure impulse.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef LOCUS_H
#define LOCUS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#define LOCUS_STACK_SIZE   64
#define LOCUS_FSTACK_SIZE  64

typedef int32_t  Cell;      // Integer cell
typedef float    FCell;     // Float cell

// Callback when resonance triggers SARTRE
typedef void (*LocusSpeakFn)(void* ctx);

// Locus Coeruleus — the resonance detector
typedef struct {
    // Data stack (integers)
    Cell   stack[LOCUS_STACK_SIZE];
    int    sp;

    // Float stack
    FCell  fstack[LOCUS_FSTACK_SIZE];
    int    fsp;

    // Return stack
    Cell   rstack[LOCUS_STACK_SIZE];
    int    rsp;

    // Vagus state pointer
    volatile void* vagus;

    // Previous geometry snapshot (for delta detection)
    float  prev_arousal;
    float  prev_coherence;
    float  prev_trauma;
    float  prev_void;

    // Callback
    LocusSpeakFn speak_fn;
    void* speak_ctx;

    // Stats
    uint64_t ticks;
    uint64_t resonances;

} Locus;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize Locus with Vagus state pointer
void locus_init(Locus* l, volatile void* vagus_state);

// Set SARTRE callback
void locus_set_speak(Locus* l, LocusSpeakFn fn, void* ctx);

// Tick — check resonance, maybe trigger SARTRE
// Returns 1 if resonance detected, 0 otherwise
int locus_tick(Locus* l);

// Execute a word by name (for testing)
int locus_exec(Locus* l, const char* word);

// Get resonance stats
uint64_t locus_get_ticks(Locus* l);
uint64_t locus_get_resonances(Locus* l);

// ═══════════════════════════════════════════════════════════════════════════════
// STACK OPERATIONS (for C integration)
// ═══════════════════════════════════════════════════════════════════════════════

void   locus_push(Locus* l, Cell val);
Cell   locus_pop(Locus* l);
void   locus_fpush(Locus* l, FCell val);
FCell  locus_fpop(Locus* l);

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY QUERY (direct C access)
// ═══════════════════════════════════════════════════════════════════════════════

// Get current field geometry metrics
float locus_geometry_pressure(Locus* l);
float locus_geometry_flow(Locus* l);
float locus_geometry_depth(Locus* l);

// Check specific patterns
int locus_is_tense(Locus* l);
int locus_is_wounded(Locus* l);
int locus_is_hollow(Locus* l);
int locus_is_prophetic(Locus* l);
int locus_is_flowing(Locus* l);

#ifdef __cplusplus
}
#endif

#endif // LOCUS_H
