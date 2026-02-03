/*
 * tongue_router.c — Multi-Model Tongue Router
 *
 * Queries SARTRE for hardware tier, resolves weight path,
 * downloads if needed, falls back gracefully.
 */

#include "tongue_router.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* ============================================================
 * STATE
 * ============================================================ */

static char router_cache_dir[512] = "tongue/weights";
static char router_resolved_path[1024] = {0};
static char router_info_buf[256] = {0};
static TongueTier router_current_tier = TONGUE_TIER_05B;
static int router_initialized = 0;

/* ============================================================
 * INTERNAL: tier → filename / URL mapping
 * ============================================================ */

static const char* tier_filename(TongueTier tier) {
    switch (tier) {
        case TONGUE_TIER_3B:  return TONGUE_3B_FILE;
        case TONGUE_TIER_15B: return TONGUE_15B_FILE;
        default:              return TONGUE_05B_FILE;
    }
}

static const char* tier_url(TongueTier tier) {
    switch (tier) {
        case TONGUE_TIER_3B:  return TONGUE_3B_URL;
        case TONGUE_TIER_15B: return TONGUE_15B_URL;
        default:              return TONGUE_05B_URL;
    }
}

/* Check if file exists and is >1MB */
static int file_exists(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0 && st.st_size > 1000000);
}

/* Download URL to path via curl. Returns 0 on success. */
static int download_weights(const char* url, const char* path) {
    /* mkdir -p */
    pid_t pid = fork();
    if (pid == 0) {
        execlp("mkdir", "mkdir", "-p", router_cache_dir, NULL);
        _exit(127);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
    }

    printf("[tongue_router] Downloading from %s...\n", url);

    pid = fork();
    if (pid == 0) {
        execlp("curl", "curl", "-L", "--progress-bar", "-o", path, url, NULL);
        _exit(127);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[tongue_router] Download failed\n");
            return -1;
        }
    } else {
        fprintf(stderr, "[tongue_router] fork() failed\n");
        return -1;
    }

    if (!file_exists(path)) {
        fprintf(stderr, "[tongue_router] Downloaded file too small or missing\n");
        return -1;
    }

    struct stat st;
    stat(path, &st);
    printf("[tongue_router] Downloaded %.1f MB → %s\n",
           st.st_size / 1024.0 / 1024.0, path);
    return 0;
}

/* Try to resolve path for a specific tier (no download) */
static const char* try_resolve(TongueTier tier) {
    snprintf(router_resolved_path, sizeof(router_resolved_path),
             "%s/%s", router_cache_dir, tier_filename(tier));
    if (file_exists(router_resolved_path)) {
        return router_resolved_path;
    }
    return NULL;
}

/* ============================================================
 * PUBLIC API
 * ============================================================ */

TongueTier tongue_router_init(const char* cache_dir) {
    if (cache_dir) {
        strncpy(router_cache_dir, cache_dir, sizeof(router_cache_dir) - 1);
        router_cache_dir[sizeof(router_cache_dir) - 1] = '\0';
    }

    router_current_tier = sartre_get_tongue_tier();
    router_initialized = 1;

    printf("[tongue_router] Initialized: tier=%s, cache=%s\n",
           sartre_tongue_tier_name(router_current_tier), router_cache_dir);
    return router_current_tier;
}

const char* tongue_router_get_weights_path(void) {
    if (!router_initialized) tongue_router_init(NULL);

    /* Try preferred tier first, then fall back */
    const char* path;

    path = try_resolve(router_current_tier);
    if (path) return path;

    /* Fall back to smaller tiers */
    if (router_current_tier >= TONGUE_TIER_3B) {
        path = try_resolve(TONGUE_TIER_15B);
        if (path) {
            fprintf(stderr, "[tongue_router] 3B not found, using 1.5B\n");
            return path;
        }
    }
    if (router_current_tier >= TONGUE_TIER_15B) {
        path = try_resolve(TONGUE_TIER_05B);
        if (path) {
            fprintf(stderr, "[tongue_router] Falling back to 0.5B\n");
            return path;
        }
    }

    return NULL;  /* nothing on disk */
}

const char* tongue_router_ensure_weights(void) {
    if (!router_initialized) tongue_router_init(NULL);

    /* Check if already on disk */
    const char* path = tongue_router_get_weights_path();
    if (path) return path;

    /* Try to download preferred tier, fall back on failure */
    TongueTier tiers[] = { router_current_tier, TONGUE_TIER_15B, TONGUE_TIER_05B };
    int n_tiers = (router_current_tier == TONGUE_TIER_3B) ? 3 :
                  (router_current_tier == TONGUE_TIER_15B) ? 2 : 1;

    for (int i = 0; i < n_tiers; i++) {
        TongueTier t = tiers[i];
        if (t > router_current_tier) continue;  /* don't try bigger than requested */

        snprintf(router_resolved_path, sizeof(router_resolved_path),
                 "%s/%s", router_cache_dir, tier_filename(t));

        if (download_weights(tier_url(t), router_resolved_path) == 0) {
            printf("[tongue_router] Ready: %s (%s)\n",
                   sartre_tongue_tier_name(t), router_resolved_path);
            return router_resolved_path;
        }

        fprintf(stderr, "[tongue_router] %s download failed, trying next tier\n",
                sartre_tongue_tier_name(t));
    }

    fprintf(stderr, "[tongue_router] All downloads failed\n");
    return NULL;
}

TongueTier tongue_router_tier(void) {
    return router_current_tier;
}

void tongue_router_set_override(TongueTier tier) {
    sartre_set_tongue_override(tier);
    router_current_tier = tier;
    router_resolved_path[0] = '\0';  /* force re-resolve */
    printf("[tongue_router] Override: %s\n", sartre_tongue_tier_name(tier));
}

void tongue_router_set_auto(void) {
    sartre_clear_tongue_override();
    router_current_tier = sartre_get_tongue_tier();
    router_resolved_path[0] = '\0';
    printf("[tongue_router] Auto: %s\n", sartre_tongue_tier_name(router_current_tier));
}

const char* tongue_router_info(void) {
    snprintf(router_info_buf, sizeof(router_info_buf),
             "tier=%s ram=%lldMB override=%s path=%s",
             sartre_tongue_tier_name(router_current_tier),
             (long long)sartre_get_total_ram_mb(),
             sartre_get_state()->tongue_override >= 0 ? "yes" : "auto",
             router_resolved_path[0] ? router_resolved_path : "(not resolved)");
    return router_info_buf;
}
