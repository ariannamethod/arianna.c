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

    // Query
    printf("[3] Querying SARTRE...\n");
    char* response = sartre_query("What is your status?");
    printf("Response: %s\n\n", response);
    free(response);

    // Shutdown
    printf("[4] Shutting down...\n");
    sartre_shutdown();
    printf("✓ SARTRE shutdown\n\n");

    printf("=== TEST COMPLETE ===\n");
    return 0;
}
