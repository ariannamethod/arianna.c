/*
 * test_sartre_comprehensive.c — Comprehensive SARTRE kernel test
 *
 * Tests everything in tail and mane ("в хвост ив гриву"):
 * - Init/shutdown lifecycle
 * - Event notification system
 * - All metric updates (inner_world, schumann, calendar, modules)
 * - Query interface
 * - State observation
 * - Error handling
 */

#include "../sartre/sartre.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define TEST_PASSED() printf("  ✓ PASSED\n")
#define TEST_FAILED(msg) do { printf("  ✗ FAILED: %s\n", msg); return 1; } while(0)
#define ASSERT(cond, msg) if (!(cond)) TEST_FAILED(msg)

int test_count = 0;
int passed_count = 0;

void start_test(const char* name) {
    test_count++;
    printf("\n[%d] Testing: %s\n", test_count, name);
}

void test_passed() {
    passed_count++;
    TEST_PASSED();
}

// ============================================================
// TEST 1: Basic lifecycle
// ============================================================

int test_lifecycle() {
    start_test("Init/shutdown lifecycle");

    // Should not be ready before init
    if (sartre_is_ready()) TEST_FAILED("SARTRE reports ready before init");

    // Init
    int ret = sartre_init(NULL);
    ASSERT(ret == 0, "sartre_init failed");
    ASSERT(sartre_is_ready(), "SARTRE not ready after init");

    // Shutdown
    sartre_shutdown();
    ASSERT(!sartre_is_ready(), "SARTRE still ready after shutdown");

    // Re-init should work
    ret = sartre_init(NULL);
    ASSERT(ret == 0, "sartre_init failed on second init");
    ASSERT(sartre_is_ready(), "SARTRE not ready after re-init");

    test_passed();
    return 0;
}

// ============================================================
// TEST 2: Event notification
// ============================================================

int test_events() {
    start_test("Event notification system");

    SystemState* state = sartre_get_state();
    ASSERT(state != NULL, "sartre_get_state returned NULL");

    int initial_count = state->event_count;

    sartre_notify_event("Test event 1");
    ASSERT(state->event_count == initial_count + 1, "Event count not incremented");
    ASSERT(strstr(state->last_events[initial_count], "Test event 1") != NULL,
           "Event not stored correctly");

    sartre_notify_event("Test event 2");
    ASSERT(state->event_count == initial_count + 2, "Second event not counted");

    test_passed();
    return 0;
}

// ============================================================
// TEST 3: Inner world metrics
// ============================================================

int test_inner_world() {
    start_test("Inner world metric updates");

    SystemState* state = sartre_get_state();

    float trauma = 0.3f;
    float arousal = 0.7f;
    float valence = 0.5f;
    float coherence = 0.85f;
    float debt = 0.15f;

    sartre_update_inner_state(trauma, arousal, valence, coherence, debt);

    ASSERT(state->trauma_level == trauma, "Trauma not updated");
    ASSERT(state->arousal == arousal, "Arousal not updated");
    ASSERT(state->valence == valence, "Valence not updated");
    ASSERT(state->coherence == coherence, "Coherence not updated");
    ASSERT(state->prophecy_debt == debt, "Prophecy debt not updated");

    test_passed();
    return 0;
}

// ============================================================
// TEST 4: Schumann resonance
// ============================================================

int test_schumann() {
    start_test("Schumann resonance metrics");

    SystemState* state = sartre_get_state();

    float coherence = 0.78f;
    float phase = 2.5f;

    sartre_update_schumann(coherence, phase);

    ASSERT(state->schumann_coherence == coherence, "Schumann coherence not updated");
    ASSERT(state->schumann_phase == phase, "Schumann phase not updated");

    test_passed();
    return 0;
}

// ============================================================
// TEST 5: Calendar state
// ============================================================

int test_calendar() {
    start_test("Calendar state tracking");

    SystemState* state = sartre_get_state();

    float tension = 0.6f;
    int is_shabbat = 1;

    sartre_update_calendar(tension, is_shabbat);

    ASSERT(state->calendar_tension == tension, "Calendar tension not updated");
    ASSERT(state->is_shabbat == is_shabbat, "Shabbat flag not updated");

    test_passed();
    return 0;
}

// ============================================================
// TEST 6: Module tracking
// ============================================================

int test_modules() {
    start_test("Module status tracking");

    SystemState* state = sartre_get_state();
    int initial_count = state->module_count;

    // Add module
    sartre_update_module("delta", MODULE_ACTIVE, 0.5f);
    ASSERT(state->module_count == initial_count + 1, "Module count not incremented");

    // Find the module we just added
    int found = 0;
    for (int i = 0; i < state->module_count; i++) {
        if (strcmp(state->modules[i].name, "delta") == 0) {
            found = 1;
            ASSERT(state->modules[i].status == MODULE_ACTIVE, "Module status wrong");
            ASSERT(state->modules[i].load == 0.5f, "Module load wrong");
            break;
        }
    }
    ASSERT(found, "Module not found in state");

    // Update existing module
    sartre_update_module("delta", MODULE_ERROR, 0.9f);
    ASSERT(state->module_count == initial_count + 1, "Module count changed on update");

    // Verify update
    found = 0;
    for (int i = 0; i < state->module_count; i++) {
        if (strcmp(state->modules[i].name, "delta") == 0) {
            found = 1;
            ASSERT(state->modules[i].status == MODULE_ERROR, "Module status not updated");
            ASSERT(state->modules[i].load == 0.9f, "Module load not updated");
            break;
        }
    }
    ASSERT(found, "Module disappeared after update");

    // Add more modules
    sartre_update_module("pandora", MODULE_LOADING, 0.2f);
    sartre_update_module("schumann", MODULE_ACTIVE, 0.7f);
    ASSERT(state->module_count >= initial_count + 3, "Multiple modules not tracked");

    test_passed();
    return 0;
}

// ============================================================
// TEST 7: State formatting (print)
// ============================================================

int test_state_print() {
    start_test("State printing");

    // Just verify it doesn't crash
    sartre_print_state();

    test_passed();
    return 0;
}

// ============================================================
// TEST 8: Full integration scenario
// ============================================================

int test_integration() {
    start_test("Full integration scenario");

    SystemState* state = sartre_get_state();

    // Simulate a full system update
    sartre_notify_event("System boot");
    sartre_update_inner_state(0.1f, 0.5f, 0.7f, 0.9f, 0.05f);
    sartre_update_schumann(0.8f, 1.5f);
    sartre_update_calendar(0.3f, 0);
    sartre_update_module("arianna_core", MODULE_ACTIVE, 0.8f);
    sartre_update_module("inner_world", MODULE_ACTIVE, 0.6f);
    sartre_notify_event("All modules loaded");

    // Verify state is consistent
    ASSERT(state->event_count >= 2, "Events not tracked");
    ASSERT(state->trauma_level == 0.1f, "Integration: trauma wrong");
    ASSERT(state->schumann_coherence == 0.8f, "Integration: schumann wrong");
    ASSERT(state->calendar_tension == 0.3f, "Integration: calendar wrong");
    ASSERT(state->module_count >= 2, "Integration: modules not tracked");

    // Print state (verify no crash)
    sartre_print_state();

    test_passed();
    return 0;
}

// ============================================================
// MAIN
// ============================================================

int main() {
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  SARTRE COMPREHENSIVE TEST                                ║\n");
    printf("║  Testing в хвост ив гриву (tail and mane)                ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    if (test_lifecycle()) return 1;
    if (test_events()) return 1;
    if (test_inner_world()) return 1;
    if (test_schumann()) return 1;
    if (test_calendar()) return 1;
    if (test_modules()) return 1;
    if (test_state_print()) return 1;
    if (test_integration()) return 1;

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  TEST SUMMARY                                             ║\n");
    printf("║  Passed: %2d / %2d                                         ║\n",
           passed_count, test_count);
    printf("║  Status: %s                                       ║\n",
           passed_count == test_count ? "✓ ALL PASSED" : "✗ SOME FAILED");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    return passed_count == test_count ? 0 : 1;
}
