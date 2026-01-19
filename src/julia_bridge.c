/*
 * julia_bridge.c — C bridge to Julia emotional engine
 * ═══════════════════════════════════════════════════════════════════════════════
 * הגשר בין סי לג'וליה
 * The bridge from C to Julia
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Architecture:
 *   - Spawns Julia subprocess with bridge.jl
 *   - Communicates via JSON over pipes
 *   - Caches results for repeated queries
 *   - Falls back gracefully if Julia not available
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include "julia_bridge.h"

/* Julia process state */
static FILE* g_julia_in = NULL;
static FILE* g_julia_out = NULL;
static pid_t g_julia_pid = 0;
static int g_julia_available = -1;  /* -1 = unknown, 0 = no, 1 = yes */

/* Path to Julia bridge script */
static char g_bridge_path[512] = "";

/* ═══════════════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void julia_shutdown(void);
int julia_is_available(void);

/* ═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int find_julia(void) {
    /* Check if Julia is installed */
    FILE* fp = popen("which julia 2>/dev/null", "r");
    if (!fp) return 0;

    char path[256];
    if (fgets(path, sizeof(path), fp) != NULL) {
        pclose(fp);
        return 1;
    }
    pclose(fp);
    return 0;
}

static void set_bridge_path(void) {
    /* Try relative path first */
    const char* paths[] = {
        "julia/bridge.jl",
        "../julia/bridge.jl",
        "./julia/bridge.jl",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], F_OK) == 0) {
            strncpy(g_bridge_path, paths[i], sizeof(g_bridge_path) - 1);
            return;
        }
    }

    /* Try from executable location */
    char exe_path[256];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        char* last_slash = strrchr(exe_path, '/');
        if (last_slash) {
            *last_slash = '\0';
            snprintf(g_bridge_path, sizeof(g_bridge_path), "%s/../julia/bridge.jl", exe_path);
            if (access(g_bridge_path, F_OK) == 0) return;
        }
    }

    /* Fallback */
    strncpy(g_bridge_path, "julia/bridge.jl", sizeof(g_bridge_path) - 1);
}

int julia_init(void) {
    if (g_julia_available == 1) return 1;  /* Already initialized */
    if (g_julia_available == 0) return 0;  /* Already failed */

    /* Check if Julia is installed */
    if (!find_julia()) {
        fprintf(stderr, "[julia] Julia not found, emotional nuances disabled\n");
        g_julia_available = 0;
        return 0;
    }

    set_bridge_path();

    if (access(g_bridge_path, F_OK) != 0) {
        fprintf(stderr, "[julia] Bridge script not found: %s\n", g_bridge_path);
        g_julia_available = 0;
        return 0;
    }

    /* Create pipes */
    int pipe_to_julia[2], pipe_from_julia[2];
    if (pipe(pipe_to_julia) < 0 || pipe(pipe_from_julia) < 0) {
        fprintf(stderr, "[julia] Failed to create pipes\n");
        g_julia_available = 0;
        return 0;
    }

    /* Fork Julia process */
    g_julia_pid = fork();
    if (g_julia_pid < 0) {
        fprintf(stderr, "[julia] Fork failed\n");
        g_julia_available = 0;
        return 0;
    }

    if (g_julia_pid == 0) {
        /* Child: become Julia */
        close(pipe_to_julia[1]);
        close(pipe_from_julia[0]);

        dup2(pipe_to_julia[0], STDIN_FILENO);
        dup2(pipe_from_julia[1], STDOUT_FILENO);

        close(pipe_to_julia[0]);
        close(pipe_from_julia[1]);

        /* Run Julia with optimizations */
        execlp("julia", "julia",
               "--startup-file=no",
               "-O1",
               g_bridge_path,
               NULL);

        /* If exec fails */
        _exit(1);
    }

    /* Parent: setup streams */
    close(pipe_to_julia[0]);
    close(pipe_from_julia[1]);

    g_julia_in = fdopen(pipe_to_julia[1], "w");
    g_julia_out = fdopen(pipe_from_julia[0], "r");

    if (!g_julia_in || !g_julia_out) {
        fprintf(stderr, "[julia] Failed to open streams\n");
        julia_shutdown();
        g_julia_available = 0;
        return 0;
    }

    /* Wait for ready signal */
    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), g_julia_out) == NULL) {
        fprintf(stderr, "[julia] No ready signal from Julia\n");
        julia_shutdown();
        g_julia_available = 0;
        return 0;
    }

    /* Check for ready */
    if (strstr(buffer, "\"ready\"") != NULL) {
        fprintf(stderr, "[julia] Emotional gradient engine ready\n");
        g_julia_available = 1;
        return 1;
    }

    fprintf(stderr, "[julia] Unexpected response: %s\n", buffer);
    julia_shutdown();
    g_julia_available = 0;
    return 0;
}

void julia_shutdown(void) {
    if (g_julia_in) {
        fprintf(g_julia_in, "{\"command\":\"quit\"}\n");
        fflush(g_julia_in);
        fclose(g_julia_in);
        g_julia_in = NULL;
    }

    if (g_julia_out) {
        fclose(g_julia_out);
        g_julia_out = NULL;
    }

    if (g_julia_pid > 0) {
        kill(g_julia_pid, SIGTERM);
        g_julia_pid = 0;
    }

    g_julia_available = -1;
}

int julia_is_available(void) {
    if (g_julia_available < 0) {
        julia_init();
    }
    return g_julia_available == 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMMUNICATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int julia_send_receive(const char* request, char* response, size_t response_size) {
    if (!julia_is_available()) return 0;

    /* Send request */
    fprintf(g_julia_in, "%s\n", request);
    fflush(g_julia_in);

    /* Read response */
    if (fgets(response, (int)response_size, g_julia_out) == NULL) {
        fprintf(stderr, "[julia] No response from Julia\n");
        return 0;
    }

    /* Remove trailing newline */
    size_t len = strlen(response);
    if (len > 0 && response[len-1] == '\n') {
        response[len-1] = '\0';
    }

    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SIMPLE JSON PARSING (no external deps)
 * ═══════════════════════════════════════════════════════════════════════════════ */

static float parse_float(const char* json, const char* key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);

    const char* pos = strstr(json, pattern);
    if (!pos) return 0.0f;

    pos += strlen(pattern);
    while (*pos == ' ' || *pos == '\t') pos++;

    return (float)atof(pos);
}

static int parse_float_array(const char* json, const char* key, float* out, int max) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":[", key);

    const char* pos = strstr(json, pattern);
    if (!pos) return 0;

    pos += strlen(pattern);

    int count = 0;
    while (count < max) {
        while (*pos == ' ' || *pos == '\t' || *pos == '\n') pos++;
        if (*pos == ']') break;

        out[count++] = (float)atof(pos);

        /* Skip to next number or end */
        while (*pos && *pos != ',' && *pos != ']') pos++;
        if (*pos == ',') pos++;
    }

    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ═══════════════════════════════════════════════════════════════════════════════ */

int julia_analyze_text(const char* text, JuliaEmotionalResult* result) {
    if (!result) return 0;
    memset(result, 0, sizeof(JuliaEmotionalResult));

    if (!julia_is_available()) return 0;

    /* Escape text for JSON */
    char escaped[1024];
    const char* s = text;
    char* d = escaped;
    while (*s && d < escaped + sizeof(escaped) - 2) {
        if (*s == '"' || *s == '\\') *d++ = '\\';
        *d++ = *s++;
    }
    *d = '\0';

    /* Build request */
    char request[2048];
    snprintf(request, sizeof(request),
             "{\"command\":\"analyze\",\"text\":\"%s\"}", escaped);

    /* Send and receive */
    char response[8192];
    if (!julia_send_receive(request, response, sizeof(response))) {
        return 0;
    }

    /* Check for error */
    if (strstr(response, "\"error\"")) {
        fprintf(stderr, "[julia] Error: %s\n", response);
        return 0;
    }

    /* Parse primary emotions */
    result->joy = parse_float(response, "joy");
    result->trust = parse_float(response, "trust");
    result->fear = parse_float(response, "fear");
    result->surprise = parse_float(response, "surprise");
    result->sadness = parse_float(response, "sadness");
    result->disgust = parse_float(response, "disgust");
    result->anger = parse_float(response, "anger");
    result->anticipation = parse_float(response, "anticipation");
    result->resonance = parse_float(response, "resonance");
    result->presence = parse_float(response, "presence");
    result->longing = parse_float(response, "longing");
    result->wonder = parse_float(response, "wonder");

    /* Parse tertiary nuances */
    result->bittersweetness = parse_float(response, "bittersweetness");
    result->nostalgia = parse_float(response, "nostalgia");
    result->serenity = parse_float(response, "serenity");
    result->melancholy = parse_float(response, "melancholy");
    result->tenderness = parse_float(response, "tenderness");
    result->vulnerability = parse_float(response, "vulnerability");
    result->wistfulness = parse_float(response, "wistfulness");
    result->euphoria = parse_float(response, "euphoria");
    result->desolation = parse_float(response, "desolation");
    result->reverence = parse_float(response, "reverence");
    result->compassion = parse_float(response, "compassion");
    result->ecstasy = parse_float(response, "ecstasy");

    return 1;
}

int julia_compute_gradient(const float* from, const float* to,
                           float* direction, float* magnitude) {
    if (!julia_is_available()) return 0;

    /* Build request */
    char request[2048];
    snprintf(request, sizeof(request),
             "{\"command\":\"gradient\","
             "\"from\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
             "\"to\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]}",
             from[0], from[1], from[2], from[3], from[4], from[5],
             from[6], from[7], from[8], from[9], from[10], from[11],
             to[0], to[1], to[2], to[3], to[4], to[5],
             to[6], to[7], to[8], to[9], to[10], to[11]);

    char response[4096];
    if (!julia_send_receive(request, response, sizeof(response))) {
        return 0;
    }

    if (strstr(response, "\"error\"")) return 0;

    parse_float_array(response, "direction", direction, 12);
    *magnitude = parse_float(response, "magnitude");

    return 1;
}

int julia_step_emotion(const float* state, const float* input, float dt,
                       float* new_state) {
    if (!julia_is_available()) return 0;

    char request[2048];
    snprintf(request, sizeof(request),
             "{\"command\":\"step\","
             "\"state\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
             "\"input\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
             "\"dt\":%.6f}",
             state[0], state[1], state[2], state[3], state[4], state[5],
             state[6], state[7], state[8], state[9], state[10], state[11],
             input[0], input[1], input[2], input[3], input[4], input[5],
             input[6], input[7], input[8], input[9], input[10], input[11],
             dt);

    char response[4096];
    if (!julia_send_receive(request, response, sizeof(response))) {
        return 0;
    }

    if (strstr(response, "\"error\"")) return 0;

    parse_float_array(response, "state", new_state, 12);
    return 1;
}

float julia_compute_resonance(const float* internal, const float* external) {
    if (!julia_is_available()) return 0.0f;

    char request[2048];
    snprintf(request, sizeof(request),
             "{\"command\":\"resonance\","
             "\"internal\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
             "\"external\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]}",
             internal[0], internal[1], internal[2], internal[3], internal[4], internal[5],
             internal[6], internal[7], internal[8], internal[9], internal[10], internal[11],
             external[0], external[1], external[2], external[3], external[4], external[5],
             external[6], external[7], external[8], external[9], external[10], external[11]);

    char response[2048];
    if (!julia_send_receive(request, response, sizeof(response))) {
        return 0.0f;
    }

    return parse_float(response, "resonance");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEBUG / PRINT
 * ═══════════════════════════════════════════════════════════════════════════════ */

void julia_print_result(const JuliaEmotionalResult* r) {
    printf("Julia Emotional Analysis:\n");
    printf("  Primary emotions (12D):\n");
    printf("    joy:          %.3f\n", r->joy);
    printf("    trust:        %.3f\n", r->trust);
    printf("    fear:         %.3f\n", r->fear);
    printf("    surprise:     %.3f\n", r->surprise);
    printf("    sadness:      %.3f\n", r->sadness);
    printf("    disgust:      %.3f\n", r->disgust);
    printf("    anger:        %.3f\n", r->anger);
    printf("    anticipation: %.3f\n", r->anticipation);
    printf("    resonance:    %.3f\n", r->resonance);
    printf("    presence:     %.3f\n", r->presence);
    printf("    longing:      %.3f\n", r->longing);
    printf("    wonder:       %.3f\n", r->wonder);
    printf("  Tertiary nuances:\n");
    printf("    bittersweetness: %.3f\n", r->bittersweetness);
    printf("    nostalgia:       %.3f\n", r->nostalgia);
    printf("    serenity:        %.3f\n", r->serenity);
    printf("    melancholy:      %.3f\n", r->melancholy);
    printf("    tenderness:      %.3f\n", r->tenderness);
    printf("    vulnerability:   %.3f\n", r->vulnerability);
    printf("    wistfulness:     %.3f\n", r->wistfulness);
    printf("    euphoria:        %.3f\n", r->euphoria);
    printf("    desolation:      %.3f\n", r->desolation);
    printf("    reverence:       %.3f\n", r->reverence);
    printf("    compassion:      %.3f\n", r->compassion);
    printf("    ecstasy:         %.3f\n", r->ecstasy);
}
