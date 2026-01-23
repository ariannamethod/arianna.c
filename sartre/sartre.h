/*
 * sartre.h — SARTRE: Linux-compatible Kernel + Metrics Hub
 *
 * SystemState aggregation + Linux compatibility layer.
 */

#ifndef SARTRE_H
#define SARTRE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// MODULE STATUS
// ═══════════════════════════════════════════════════════════════════════════════

typedef enum {
    MODULE_UNKNOWN = 0,
    MODULE_IDLE,
    MODULE_ACTIVE,
    MODULE_ERROR,
    MODULE_LOADING,
    MODULE_UNLOADING
} ModuleStatus;

// ═══════════════════════════════════════════════════════════════════════════════
// MODULE INFO
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char name[64];
    ModuleStatus status;
    float load;              // 0-1: resource usage
    int64_t last_active_ms;  // timestamp
    char last_event[128];    // last event description
} ModuleInfo;

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM STATE (central metrics hub)
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    // Modules
    ModuleInfo modules[16];
    int module_count;

    // Resources
    float memory_pressure;   // 0-1
    float cpu_load;          // 0-1

    // Inner world (from Go goroutines)
    float trauma_level;
    float arousal;
    float valence;
    float coherence;
    float prophecy_debt;
    float entropy;

    // Schumann
    float schumann_coherence;
    float schumann_phase;

    // Calendar
    float calendar_tension;
    int is_shabbat;

    // Flags
    int spiral_detected;
    int wormhole_active;
    int strange_loop;

    // Last events
    char last_events[8][256];
    int event_count;
} SystemState;

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

int sartre_init(const char* config_path);
void sartre_shutdown(void);
int sartre_is_ready(void);

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC UPDATES
// ═══════════════════════════════════════════════════════════════════════════════

void sartre_notify_event(const char* event);
void sartre_update_inner_state(float trauma, float arousal, float valence,
                                float coherence, float prophecy_debt);
void sartre_update_schumann(float coherence, float phase);
void sartre_update_calendar(float tension, int is_shabbat);
void sartre_update_module(const char* name, ModuleStatus status, float load);
void sartre_update_state(SystemState* state);
SystemState* sartre_get_state(void);

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG / MONITORING
// ═══════════════════════════════════════════════════════════════════════════════

void sartre_print_state(void);

#ifdef __cplusplus
}
#endif

#endif // SARTRE_H
