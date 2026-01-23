/*
 * test_sartre.c — Quick test for SARTRE kernel
 */

#include "sartre.h"
#include <stdio.h>

int main() {
    printf("=== SARTRE KERNEL TEST ===\n\n");

    // Init
    printf("[1] Initializing SARTRE...\n");
    if (sartre_init(NULL) != 0) {
        fprintf(stderr, "SARTRE init failed\n");
        return 1;
    }
    printf("✓ SARTRE initialized\n\n");

    // Notify event
    printf("[2] Notifying event...\n");
    sartre_notify_event("Package numpy connected");
    printf("✓ Event notified\n\n");

    // Update state
    printf("[3] Updating state...\n");
    sartre_update_inner_state(0.2f, 0.6f, 0.7f, 0.85f, 0.1f);
    sartre_update_schumann(0.8f, 1.2f);
    sartre_update_calendar(0.3f, 0);
    sartre_update_module("arianna_core", MODULE_ACTIVE, 0.7f);
    printf("✓ State updated\n\n");

    // Print state
    printf("[4] Printing state...\n\n");
    sartre_print_state();
    printf("✓ State printed\n\n");

    // Shutdown
    printf("[5] Shutting down...\n");
    sartre_shutdown();
    printf("✓ SARTRE shutdown\n\n");

    printf("=== TEST COMPLETE ===\n");
    return 0;
}
