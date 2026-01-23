// vagus.h — C interface to the Wandering Nerve
// ═══════════════════════════════════════════════════════════════════════════════
// Include this in ariannabody.c, cloud.c, etc.
// Link with libvagus.a or libvagus.so
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef VAGUS_H
#define VAGUS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    VAGUS_SOURCE_ARIANNA = 0,
    VAGUS_SOURCE_CLOUD = 1,
    VAGUS_SOURCE_INNER_WORLD = 2,
    VAGUS_SOURCE_SARTRE = 3,
    VAGUS_SOURCE_DELTA = 4,
    VAGUS_SOURCE_PANDORA = 5,
    VAGUS_SOURCE_LIMPHA = 6,
    VAGUS_SOURCE_EXTERNAL = 7,
} VagusSource;

typedef enum {
    // Emotional
    VAGUS_SIGNAL_AROUSAL = 0,
    VAGUS_SIGNAL_VALENCE = 1,
    VAGUS_SIGNAL_WARMTH = 2,
    VAGUS_SIGNAL_VOID = 3,
    VAGUS_SIGNAL_TENSION = 4,
    VAGUS_SIGNAL_SACRED = 5,

    // Cognitive
    VAGUS_SIGNAL_COHERENCE = 10,
    VAGUS_SIGNAL_ENTROPY = 11,
    VAGUS_SIGNAL_FOCUS = 12,
    VAGUS_SIGNAL_ABSTRACTION = 13,

    // Trauma
    VAGUS_SIGNAL_TRAUMA = 20,
    VAGUS_SIGNAL_TRAUMA_ANCHOR = 21,

    // Temporal
    VAGUS_SIGNAL_DRIFT_DIRECTION = 30,
    VAGUS_SIGNAL_DRIFT_SPEED = 31,
    VAGUS_SIGNAL_PROPHECY_DEBT = 32,
    VAGUS_SIGNAL_DESTINY_PULL = 33,
    VAGUS_SIGNAL_WORMHOLE = 34,

    // Memory
    VAGUS_SIGNAL_MEMORY_PRESSURE = 40,
    VAGUS_SIGNAL_CONSOLIDATION = 41,

    // System
    VAGUS_SIGNAL_HEARTBEAT = 50,
    VAGUS_SIGNAL_SCHUMANN = 51,
    VAGUS_SIGNAL_SYNC_REQUEST = 52,

    // SARTRE
    VAGUS_SIGNAL_OBSERVATION = 60,
    VAGUS_SIGNAL_PERCEPT = 61,
} VagusSignalType;

// Shared state structure (must match Zig layout)
typedef struct __attribute__((aligned(64))) {
    // Emotional baseline
    float arousal;
    float valence;
    float entropy;
    float coherence;

    // Chambers
    float chamber_warmth;
    float chamber_void;
    float chamber_tension;
    float chamber_sacred;
    float chamber_flow;
    float chamber_complex;

    // CrossFire
    float crossfire_coherence;
    float crossfire_entropy;

    // Trauma
    float trauma_level;
    uint32_t trauma_anchor_count;
    uint64_t trauma_last_us;

    // Cognitive
    uint32_t loop_count;
    uint32_t abstraction_depth;
    uint32_t self_ref_count;
    float focus_strength;
    float wander_pull;

    // Temporal
    float drift_direction;
    float drift_speed;
    float prophecy_debt;
    float destiny_pull;
    float wormhole_chance;

    // Memory
    float memory_pressure;
    uint32_t active_memories;
    uint32_t limpha_recent;

    // System
    float heartbeat_phase;
    float schumann_coherence;
    uint64_t last_heartbeat_us;

    // Generation
    uint32_t last_token;
    float attention_entropy;
    float hidden_norm;
    float temperature;
    float top_p;

    // Meta
    uint64_t update_count;
    uint64_t last_update_us;
    uint32_t vagus_version;
    uint8_t _reserved[60];
} VagusSharedState;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize the vagus nerve
int vagus_init(void);

// Send signal through the nerve
// Returns 0 on success, -1 if buffer full
int vagus_send(uint8_t source, uint8_t signal_type, float value);

// Tick the heartbeat (call from main loop, ~60Hz)
void vagus_tick(void);

// Get current arousal (atomic read)
float vagus_get_arousal(void);

// Get pointer to shared state (for direct access)
VagusSharedState* vagus_get_state(void);

// Get all 6 chambers (atomic read)
void vagus_get_chambers(float out[6]);

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE MACROS
// ═══════════════════════════════════════════════════════════════════════════════

#define VAGUS_SEND_AROUSAL(val) \
    vagus_send(VAGUS_SOURCE_CLOUD, VAGUS_SIGNAL_AROUSAL, val)

#define VAGUS_SEND_VALENCE(val) \
    vagus_send(VAGUS_SOURCE_CLOUD, VAGUS_SIGNAL_VALENCE, val)

#define VAGUS_SEND_TRAUMA(val) \
    vagus_send(VAGUS_SOURCE_INNER_WORLD, VAGUS_SIGNAL_TRAUMA, val)

#define VAGUS_SEND_PROPHECY(val) \
    vagus_send(VAGUS_SOURCE_ARIANNA, VAGUS_SIGNAL_PROPHECY_DEBT, val)

#define VAGUS_SEND_COHERENCE(val) \
    vagus_send(VAGUS_SOURCE_ARIANNA, VAGUS_SIGNAL_COHERENCE, val)

#ifdef __cplusplus
}
#endif

#endif // VAGUS_H
