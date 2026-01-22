/*
 * cloud_go.h — Cloud 200K Pre-Processor (Go implementation)
 *
 * "Something fires BEFORE meaning arrives"
 *
 * This is the interface to the Go-based Cloud preprocessor.
 * Cloud detects emotional tone BEFORE the main model processes text.
 */

#ifndef CLOUD_GO_H
#define CLOUD_GO_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Cloud with weights directory
// Returns 0 on success
int cloud_init(const char* weights_dir);

// Preprocess text (async internally, blocking externally)
// Returns number of cross-fire iterations
int cloud_preprocess(const char* text);

// Get temperature bias from last preprocessing
// Range: [-0.2, +0.2]
float cloud_get_temperature_bias(void);

// Get primary emotion word from last preprocessing
const char* cloud_get_primary(void);

// Get secondary emotion word from last preprocessing
const char* cloud_get_secondary(void);

// Get chamber activation by name (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX)
float cloud_get_chamber(const char* name);

// Full ping (returns formatted string: primary|secondary|iterations|temp_bias|chambers)
const char* cloud_ping(const char* text);

// Stop async preprocessor
void cloud_stop(void);

// Free Cloud resources
void cloud_free(void);

#ifdef __cplusplus
}
#endif

#endif // CLOUD_GO_H
