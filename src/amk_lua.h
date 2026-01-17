/*
 * amk_lua.h - Lua scripting layer for AMK (Arianna Method Kernel)
 *
 * Hot-reload Lua scripts that control:
 * - Field dynamics (prophecy, destiny, wormhole)
 * - Suffering parameters (pain, tension, dissonance)
 * - Velocity modes (walk, run, backward)
 * - Pack management (CODES/RIC, DarkMatter, NoTorch)
 *
 * "Movement IS language. Lua IS movement at runtime."
 */

#ifndef AMK_LUA_H
#define AMK_LUA_H

#ifdef USE_LUA

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize Lua VM with AMK bindings
// Returns 0 on success, -1 on failure
int amk_lua_init(void);

// Cleanup Lua VM
void amk_lua_shutdown(void);

// Get Lua state (for advanced usage)
lua_State* amk_lua_state(void);

// ═══════════════════════════════════════════════════════════════════════════════
// SCRIPT EXECUTION
// ═══════════════════════════════════════════════════════════════════════════════

// Load and execute Lua script file
// Returns 0 on success, -1 on failure
int amk_lua_load(const char* path);

// Execute Lua string directly
// Returns 0 on success, -1 on failure
int amk_lua_exec(const char* code);

// Reload last loaded script (hot-reload)
// Returns 0 on success, -1 on failure
int amk_lua_reload(void);

// ═══════════════════════════════════════════════════════════════════════════════
// CALLBACKS (called from C at specific points)
// ═══════════════════════════════════════════════════════════════════════════════

// Called before generation (Lua can modify state)
void amk_lua_on_generate_start(const char* prompt, int max_tokens, float temperature);

// Called after generation
void amk_lua_on_generate_end(const char* output, int tokens_generated);

// Called on each token (can trigger wormholes, adjust params)
void amk_lua_on_token(int token_id, float prob, int position);

// Called when trauma detected
void amk_lua_on_trauma(float intensity, const char* trigger);

// Called on emotional drift
void amk_lua_on_emotion(float valence, float arousal);

// ═══════════════════════════════════════════════════════════════════════════════
// LUA -> C QUERY FUNCTIONS (Lua can read AMK state)
// ═══════════════════════════════════════════════════════════════════════════════

// These are registered as Lua functions automatically:
//
// amk.prophecy()        -> int
// amk.destiny()         -> float
// amk.wormhole()        -> float
// amk.pain()            -> float
// amk.tension()         -> float
// amk.dissonance()      -> float
// amk.velocity()        -> int (0=nomove, 1=walk, 2=run, -1=backward)
// amk.effective_temp()  -> float
// amk.debt()            -> float
// amk.pack_enabled(name) -> bool

// ═══════════════════════════════════════════════════════════════════════════════
// LUA -> C COMMAND FUNCTIONS (Lua can modify AMK state)
// ═══════════════════════════════════════════════════════════════════════════════

// These are registered as Lua functions automatically:
//
// amk.set_prophecy(n)      -- set prophecy depth 1-64
// amk.set_destiny(f)       -- set destiny pull 0-1
// amk.set_wormhole(f)      -- set wormhole chance 0-1
// amk.set_pain(f)          -- set pain 0-1
// amk.set_tension(f)       -- set tension 0-1
// amk.set_dissonance(f)    -- set dissonance 0-1
// amk.set_velocity(mode)   -- "walk", "run", "nomove", "backward"
// amk.jump(n)              -- queue jump of n tokens
// amk.exec(script)         -- execute raw DSL script
// amk.reset_field()        -- reset suffering state
// amk.reset_debt()         -- reset debt
// amk.enable_pack(name)    -- enable pack by name
// amk.disable_pack(name)   -- disable pack by name

#else /* !USE_LUA */

// Stubs when Lua is not compiled in
static inline int amk_lua_init(void) { return 0; }
static inline void amk_lua_shutdown(void) {}
static inline int amk_lua_load(const char* path) { (void)path; return -1; }
static inline int amk_lua_exec(const char* code) { (void)code; return -1; }
static inline int amk_lua_reload(void) { return -1; }
static inline void amk_lua_on_generate_start(const char* p, int m, float t) { (void)p; (void)m; (void)t; }
static inline void amk_lua_on_generate_end(const char* o, int t) { (void)o; (void)t; }
static inline void amk_lua_on_token(int t, float p, int pos) { (void)t; (void)p; (void)pos; }
static inline void amk_lua_on_trauma(float i, const char* t) { (void)i; (void)t; }
static inline void amk_lua_on_emotion(float v, float a) { (void)v; (void)a; }

#endif /* USE_LUA */

#endif /* AMK_LUA_H */
