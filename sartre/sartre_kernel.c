/*
 * sartre_kernel.c — SARTRE: Linux-compatible Kernel + Metrics Hub
 *
 * SystemState aggregation + Linux compatibility layer.
 */

#include "sartre.h"
#include <time.h>
#include <string.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

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
    system_state.tongue_override = -1;  // auto mode
    sartre_initialized = 1;

    sartre_detect_tongue_tier();

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
        // Shift events (use strncpy to prevent buffer overflow)
        for (int i = 0; i < 7; i++) {
            strncpy(system_state.last_events[i], system_state.last_events[i+1], 255);
            system_state.last_events[i][255] = '\0';
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

    printf("Tongue:\n");
    printf("  tier: %s\n", sartre_tongue_tier_name(system_state.tongue_tier));
    printf("  ram: %lld MB\n", (long long)system_state.total_ram_mb);
    printf("  override: %s\n", system_state.tongue_override >= 0 ? "yes" : "auto");
    printf("\n");
}

// ============================================================
// TONGUE ROUTING (hardware-based model selection)
// ============================================================

static int64_t detect_total_ram_mb(void) {
#ifdef __APPLE__
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return memsize / (1024 * 1024);
    }
    return 4096;
#else
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        int64_t total_kb = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                sscanf(line + 9, " %lld", &total_kb);
                break;
            }
        }
        fclose(f);
        if (total_kb > 0) return total_kb / 1024;
    }
    return 4096;
#endif
}

TongueTier sartre_detect_tongue_tier(void) {
    int64_t ram_mb = detect_total_ram_mb();
    system_state.total_ram_mb = ram_mb;

    if (system_state.tongue_override >= 0) {
        system_state.tongue_tier = (TongueTier)system_state.tongue_override;
        fprintf(stderr, "[sartre] RAM: %lld MB | tongue tier: %s (override)\n",
                (long long)ram_mb, sartre_tongue_tier_name(system_state.tongue_tier));
        return system_state.tongue_tier;
    }

    /* Conservative thresholds:
     * 3B needs ~2.8GB runtime, want 60%+ free → 8GB+
     * 1.5B needs ~1.4GB → 4GB+
     * 0.5B needs ~537MB → anything */
    if (ram_mb >= 8000) {
        system_state.tongue_tier = TONGUE_TIER_3B;
    } else if (ram_mb >= 4000) {
        system_state.tongue_tier = TONGUE_TIER_15B;
    } else {
        system_state.tongue_tier = TONGUE_TIER_05B;
    }

    fprintf(stderr, "[sartre] RAM: %lld MB | tongue tier: %s (auto)\n",
            (long long)ram_mb, sartre_tongue_tier_name(system_state.tongue_tier));
    return system_state.tongue_tier;
}

void sartre_set_tongue_override(TongueTier tier) {
    system_state.tongue_override = (int)tier;
    system_state.tongue_tier = tier;
    fprintf(stderr, "[sartre] tongue override: %s\n", sartre_tongue_tier_name(tier));
}

void sartre_clear_tongue_override(void) {
    system_state.tongue_override = -1;
    sartre_detect_tongue_tier();
}

TongueTier sartre_get_tongue_tier(void) {
    return system_state.tongue_tier;
}

const char* sartre_tongue_tier_name(TongueTier tier) {
    switch (tier) {
        case TONGUE_TIER_05B: return "0.5B";
        case TONGUE_TIER_15B: return "1.5B";
        case TONGUE_TIER_3B:  return "3B";
        default: return "unknown";
    }
}

int64_t sartre_get_total_ram_mb(void) {
    return system_state.total_ram_mb;
}
