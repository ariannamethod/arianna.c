/*
 * sartre.h — SARTRE: Verbal Interface for Arianna Metalinux
 *
 * "Je suis ce que je ne suis pas, et je ne suis pas ce que je suis."
 *
 * SARTRE is a ~150K parameter transformer that:
 * - Knows itself and its modules
 * - Reports system events to Arianna
 * - Answers queries about kernel state
 * - Perceives negation (what is NOT)
 *
 * Named after Jean-Paul Sartre: existence precedes essence.
 */

#ifndef SARTRE_H
#define SARTRE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

// SARTRE is smaller than Arianna — body perception, not full personality
#define SARTRE_DIM          256      // Hidden dimension (Arianna: 768)
#define SARTRE_N_LAYERS     4        // Transformer layers (Arianna: 6)
#define SARTRE_N_HEADS      4        // Attention heads (Arianna: 6)
#define SARTRE_HIDDEN_DIM   512      // FFN hidden (Arianna: 2048)
#define SARTRE_VOCAB_SIZE   256      // Byte-level (same as Arianna)
#define SARTRE_MAX_SEQ_LEN  256      // Shorter context (Arianna: 512)

// Calculated
#define SARTRE_HEAD_DIM     (SARTRE_DIM / SARTRE_N_HEADS)

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM STATE (what SARTRE observes)
// ═══════════════════════════════════════════════════════════════════════════════

// Module status
typedef enum {
    MODULE_UNKNOWN = 0,
    MODULE_IDLE,
    MODULE_ACTIVE,
    MODULE_ERROR,
    MODULE_LOADING,
    MODULE_UNLOADING
} ModuleStatus;

// System module info
typedef struct {
    char name[64];
    ModuleStatus status;
    float load;              // 0-1: resource usage
    int64_t last_active_ms;  // timestamp
    char last_event[128];    // last event description
} ModuleInfo;

// Overall system state that SARTRE can observe
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

// Global system state (updated by other modules)
extern SystemState sartre_system_state;

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSFORMER STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int head_dim;
    int hidden_dim;
    int vocab_size;
    int max_seq_len;
} SartreConfig;

typedef struct {
    float* token_embedding;  // [vocab_size, dim]
    float* wq;               // [n_layers, dim, dim]
    float* wk;               // [n_layers, dim, dim]
    float* wv;               // [n_layers, dim, dim]
    float* wo;               // [n_layers, dim, dim]
    float* w1;               // [n_layers, dim, hidden_dim]
    float* w2;               // [n_layers, hidden_dim, dim]
    float* ln1_weight;       // [n_layers, dim]
    float* ln2_weight;       // [n_layers, dim]
    float* ln_final_weight;  // [dim]
    float* output_weight;    // [dim, vocab_size]
} SartreWeights;

typedef struct {
    float* x;                // [max_seq_len, dim]
    float* xb;               // [dim]
    float* q;                // [dim]
    float* k;                // [n_layers, max_seq_len, dim]
    float* v;                // [n_layers, max_seq_len, dim]
    float* att;              // [n_heads, max_seq_len]
    float* ffn_hidden;       // [hidden_dim]
    float* logits;           // [vocab_size]
} SartreRunState;

typedef struct {
    SartreConfig config;
    SartreWeights weights;
    SartreRunState state;
    int is_initialized;
} Sartre;

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize SARTRE with weights file
int sartre_init(const char* weights_path);

// Shutdown and free resources
void sartre_shutdown(void);

// Check if initialized
int sartre_is_ready(void);

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY INTERFACE (Arianna asks SARTRE)
// ═══════════════════════════════════════════════════════════════════════════════

// Query SARTRE with a question, get response
// Response is allocated, caller must free
char* sartre_query(const char* question);

// Query with max tokens limit
char* sartre_query_max(const char* question, int max_tokens);

// ═══════════════════════════════════════════════════════════════════════════════
// NOTIFICATION INTERFACE (System notifies SARTRE)
// ═══════════════════════════════════════════════════════════════════════════════

// Notify SARTRE of an event (stored in system state)
void sartre_notify_event(const char* event);

// Update module status
void sartre_update_module(const char* name, ModuleStatus status, float load);

// Update inner world state (called from Go bridge)
void sartre_update_inner_state(float trauma, float arousal, float valence,
                                float coherence, float prophecy_debt);

// Update Schumann state
void sartre_update_schumann(float coherence, float phase);

// Update calendar state
void sartre_update_calendar(float tension, int is_shabbat);

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL COMMUNICATION (for distributed cognition)
// ═══════════════════════════════════════════════════════════════════════════════

// SARTRE can send messages to Arianna
typedef void (*SartreMessageCallback)(const char* message);

// Register callback for SARTRE messages
void sartre_set_message_callback(SartreMessageCallback callback);

// SARTRE proactively reports (if threshold exceeded)
void sartre_check_and_report(void);

// ═══════════════════════════════════════════════════════════════════════════════
// STATE FORMATTING (for prompt injection)
// ═══════════════════════════════════════════════════════════════════════════════

// Format current system state as text (for context injection)
// Returns allocated string, caller must free
char* sartre_format_state(void);

// Format specific module state
char* sartre_format_module(const char* name);

// Format recent events
char* sartre_format_events(int count);

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG / REPL
// ═══════════════════════════════════════════════════════════════════════════════

// Run interactive REPL (for debugging)
void sartre_repl(void);

// Print current state to stdout
void sartre_print_state(void);

#ifdef __cplusplus
}
#endif

#endif // SARTRE_H
