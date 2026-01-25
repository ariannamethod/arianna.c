// pandora_bridge.c — Bridge to external brains (GPT2-30M, TinyLlama GGUF)
// Part of SARTRE kernel package system
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "pandora_bridge.h"

// Store child PID for cleanup
static pid_t g_last_brain_pid = -1;

// ═══════════════════════════════════════════════════════════════════════════════
// BRAIN TYPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

const char* external_brain_name(ExternalBrainType brain) {
    switch (brain) {
        case BRAIN_GPT2_30M:    return "GPT2-30M";
        case BRAIN_TINYLLAMA:   return "TinyLlama-1.1B";
        case BRAIN_GPT2_DISTILL: return "GPT2-distill";
        default:                return "Unknown";
    }
}

static const char* brain_script(ExternalBrainType brain) {
    switch (brain) {
        case BRAIN_GPT2_30M:    return EXTERNAL_BRAIN_GPT2_SCRIPT;
        case BRAIN_TINYLLAMA:   return EXTERNAL_BRAIN_GGUF_SCRIPT;
        case BRAIN_GPT2_DISTILL: return EXTERNAL_BRAIN_TORCH_SCRIPT;
        default:                return EXTERNAL_BRAIN_GPT2_SCRIPT;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECURITY: Sanitize prompt to prevent command injection
// ═══════════════════════════════════════════════════════════════════════════════

static void sanitize_prompt(const char* input, char* output, size_t max_len) {
    size_t j = 0;
    for (size_t i = 0; input[i] && j < max_len - 1; i++) {
        char c = input[i];
        // Allow only safe characters: alphanumeric, space, basic punctuation
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == ' ' || c == '.' ||
            c == ',' || c == '?' || c == '!' || c == '-' || c == ':') {
            output[j++] = c;
        }
        // Skip all other characters (quotes, backticks, $, etc.)
    }
    output[j] = '\0';
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTERNAL BRAIN EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════

int external_brain_extract_from(ExternalBrainType brain, const char* prompt, int* tokens, int max_tokens) {
    // Sanitize prompt to prevent injection
    char safe_prompt[512];
    sanitize_prompt(prompt, safe_prompt, sizeof(safe_prompt));

    if (strlen(safe_prompt) == 0) {
        fprintf(stderr, "[pandora_bridge] Empty prompt after sanitization\n");
        return -1;
    }

    // SECURITY: Use pipe+fork+exec instead of popen to avoid shell
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        fprintf(stderr, "[pandora_bridge] pipe() failed\n");
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        close(pipefd[0]);
        close(pipefd[1]);
        fprintf(stderr, "[pandora_bridge] fork() failed\n");
        return -1;
    }

    if (pid == 0) {
        // Child: redirect stdout to pipe, exec python directly
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        // Redirect stderr to /dev/null
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }

        // exec with argv array - no shell!
        execlp("python3", "python3", brain_script(brain), safe_prompt, "50", "--tokens", (char*)NULL);
        _exit(127);  // exec failed
    }

    // Parent: read from pipe
    close(pipefd[1]);
    FILE* fp = fdopen(pipefd[0], "r");
    if (!fp) {
        close(pipefd[0]);
        waitpid(pid, NULL, 0);
        fprintf(stderr, "[pandora_bridge] fdopen() failed\n");
        return -1;
    }

    // Read output: FORMAT is "COUNT:tok1,tok2,tok3,..."
    char output[8192];
    if (!fgets(output, sizeof(output), fp)) {
        fclose(fp);
        waitpid(pid, NULL, 0);
        fprintf(stderr, "[pandora_bridge] No output from external brain\n");
        return -1;
    }
    fclose(fp);
    waitpid(pid, NULL, 0);  // Reap child process

    // Parse count
    int count = 0;
    char* colon = strchr(output, ':');
    if (!colon) {
        fprintf(stderr, "[pandora_bridge] Invalid output format\n");
        return -1;
    }

    count = atoi(output);
    if (count <= 0 || count > max_tokens) {
        fprintf(stderr, "[pandora_bridge] Invalid token count: %d\n", count);
        return -1;
    }

    // Parse tokens
    char* tok_str = colon + 1;
    int n_parsed = 0;
    char* token = strtok(tok_str, ",\n");

    while (token && n_parsed < count && n_parsed < max_tokens) {
        tokens[n_parsed++] = atoi(token);
        token = strtok(NULL, ",\n");
    }

    return n_parsed;
}

// Default: use GPT2-30M
int external_brain_extract(const char* prompt, int* tokens, int max_tokens) {
    return external_brain_extract_from(BRAIN_GPT2_30M, prompt, tokens, max_tokens);
}


// ═══════════════════════════════════════════════════════════════════════════════
// PANDORA INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

int pandora_steal_from(PandoraBox* pandora, ExternalBrainType brain, const char* prompt) {
    int tokens[EXTERNAL_BRAIN_MAX_TOKENS];

    printf("[pandora] Extracting vocabulary from %s...\n", external_brain_name(brain));
    printf("[pandora] Prompt: \"%s\"\n", prompt);

    int n_tokens = external_brain_extract_from(brain, prompt, tokens, EXTERNAL_BRAIN_MAX_TOKENS);

    if (n_tokens <= 0) {
        printf("[pandora] Failed to extract from external brain\n");
        return 0;
    }

    printf("[pandora] Received %d Arianna tokens from %s\n", n_tokens, external_brain_name(brain));

    // Feed to Pandora - extract n-grams (1-3)
    int old_count = pandora->n_ngrams;
    pandora_extract(pandora, tokens, n_tokens, 1, 3);
    int new_ngrams = pandora->n_ngrams - old_count;

    printf("[pandora] Extracted %d new n-grams (total: %d)\n",
           new_ngrams, pandora->n_ngrams);

    return new_ngrams;
}

// Default: use GPT2-30M
int pandora_steal_from_brain(PandoraBox* pandora, const char* prompt) {
    return pandora_steal_from(pandora, BRAIN_GPT2_30M, prompt);
}
