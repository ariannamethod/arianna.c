/*
 * sartre_kernel.c — SARTRE: Linux-compatible Kernel + Metrics Hub
 *
 * SystemState aggregation + Linux compatibility layer.
 */

#include "sartre.h"
#include <time.h>
#include <string.h>

// ============================================================
// GLOBAL STATE
// ============================================================

static SystemState system_state = {0};
static int sartre_initialized = 0;

// ============================================================
// LIFECYCLE
// ============================================================

int sartre_init(const char* config_path) {
    (void)config_path; // unused for now

    memset(&system_state, 0, sizeof(SystemState));
    sartre_initialized = 1;

    fprintf(stderr, "[sartre] kernel initialized\n");
    return 0;
}

void sartre_shutdown(void) {
    if (!sartre_initialized) return;
    sartre_initialized = 0;
    fprintf(stderr, "[sartre] kernel shutdown\n");
}

int sartre_is_ready(void) {
    return sartre_initialized;
}

// ============================================================
// METRIC UPDATES
// ============================================================

void sartre_notify_event(const char* event) {
    if (!sartre_initialized || !event) return;

    if (system_state.event_count < 8) {
        strncpy(system_state.last_events[system_state.event_count],
                event, 255);
        system_state.last_events[system_state.event_count][255] = '\0';
        system_state.event_count++;
    } else {
        // Shift events
        for (int i = 0; i < 7; i++) {
            strcpy(system_state.last_events[i], system_state.last_events[i+1]);
        }
        strncpy(system_state.last_events[7], event, 255);
        system_state.last_events[7][255] = '\0';
    }

    fprintf(stderr, "[sartre] event: %s\n", event);
}

void sartre_update_inner_state(float trauma, float arousal, float valence,
                                 float coherence, float prophecy_debt) {
    if (!sartre_initialized) return;

    system_state.trauma_level = trauma;
    system_state.arousal = arousal;
    system_state.valence = valence;
    system_state.coherence = coherence;
    system_state.prophecy_debt = prophecy_debt;
}

void sartre_update_schumann(float coherence, float phase) {
    if (!sartre_initialized) return;

    system_state.schumann_coherence = coherence;
    system_state.schumann_phase = phase;
}

void sartre_update_calendar(float tension, int is_shabbat) {
    if (!sartre_initialized) return;

    system_state.calendar_tension = tension;
    system_state.is_shabbat = is_shabbat;
}

void sartre_update_module(const char* name, ModuleStatus status, float load) {
    if (!sartre_initialized || !name) return;

    // Find existing or add new
    int idx = -1;
    for (int i = 0; i < system_state.module_count; i++) {
        if (strncmp(system_state.modules[i].name, name, 63) == 0) {
            idx = i;
            break;
        }
    }

    if (idx == -1 && system_state.module_count < 16) {
        idx = system_state.module_count++;
        strncpy(system_state.modules[idx].name, name, 63);
        system_state.modules[idx].name[63] = '\0';
    }

    if (idx >= 0) {
        system_state.modules[idx].status = status;
        system_state.modules[idx].load = load;
        system_state.modules[idx].last_active_ms = (int64_t)time(NULL) * 1000;
    }
}

void sartre_update_state(SystemState* state) {
    if (state) {
        memcpy(&system_state, state, sizeof(SystemState));
    }
}

SystemState* sartre_get_state(void) {
    return &system_state;
}

// ============================================================
// DEBUG / MONITORING
// ============================================================

void sartre_print_state(void) {
    if (!sartre_initialized) {
        printf("[sartre] not initialized\n");
        return;
    }

    printf("\n=== SARTRE KERNEL STATE ===\n\n");

    printf("Inner World:\n");
    printf("  trauma_level: %.2f\n", system_state.trauma_level);
    printf("  arousal: %.2f\n", system_state.arousal);
    printf("  valence: %.2f\n", system_state.valence);
    printf("  coherence: %.2f\n", system_state.coherence);
    printf("  prophecy_debt: %.2f\n", system_state.prophecy_debt);
    printf("  entropy: %.2f\n\n", system_state.entropy);

    printf("Schumann:\n");
    printf("  coherence: %.2f\n", system_state.schumann_coherence);
    printf("  phase: %.2f\n\n", system_state.schumann_phase);

    printf("Calendar:\n");
    printf("  tension: %.2f\n", system_state.calendar_tension);
    printf("  is_shabbat: %d\n\n", system_state.is_shabbat);

    printf("Resources:\n");
    printf("  memory_pressure: %.2f\n", system_state.memory_pressure);
    printf("  cpu_load: %.2f\n\n", system_state.cpu_load);

    printf("Modules (%d):\n", system_state.module_count);
    for (int i = 0; i < system_state.module_count; i++) {
        printf("  [%d] %s: status=%d load=%.2f\n",
               i, system_state.modules[i].name,
               system_state.modules[i].status,
               system_state.modules[i].load);
    }
    printf("\n");

    printf("Recent Events (%d):\n", system_state.event_count);
    for (int i = 0; i < system_state.event_count; i++) {
        printf("  [%d] %s\n", i, system_state.last_events[i]);
    }
    printf("\n");

    printf("Flags:\n");
    printf("  spiral_detected: %d\n", system_state.spiral_detected);
    printf("  wormhole_active: %d\n", system_state.wormhole_active);
    printf("  strange_loop: %d\n", system_state.strange_loop);
    printf("\n");
}
