/*
 * amk_lua.c - Lua scripting layer for AMK (Arianna Method Kernel)
 *
 * Hot-reload Lua scripts at runtime without recompiling C.
 * "The field IS the script. The script IS the field."
 *
 * Build with: gcc -DUSE_LUA ... -llua
 */

#ifdef USE_LUA

#include "amk_lua.h"
#include "amk_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════

static lua_State* L = NULL;
static char g_last_script_path[512] = {0};
static int g_initialized = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// LUA ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

static void lua_error_handler(lua_State* L, const char* context) {
    const char* msg = lua_tostring(L, -1);
    fprintf(stderr, "[AMK Lua] %s error: %s\n", context, msg ? msg : "(unknown)");
    lua_pop(L, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUA -> C: QUERY FUNCTIONS (read AMK state)
// ═══════════════════════════════════════════════════════════════════════════════

static int l_amk_prophecy(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushinteger(L, s->prophecy);
    return 1;
}

static int l_amk_destiny(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->destiny);
    return 1;
}

static int l_amk_wormhole(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->wormhole);
    return 1;
}

static int l_amk_pain(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->pain);
    return 1;
}

static int l_amk_tension(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->tension);
    return 1;
}

static int l_amk_dissonance(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->dissonance);
    return 1;
}

static int l_amk_velocity(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushinteger(L, s->velocity_mode);
    return 1;
}

static int l_amk_effective_temp(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->effective_temp);
    return 1;
}

static int l_amk_debt(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->debt);
    return 1;
}

static int l_amk_temporal_debt(lua_State* L) {
    AM_State* s = am_get_state();
    lua_pushnumber(L, s->temporal_debt);
    return 1;
}

static int l_amk_pack_enabled(lua_State* L) {
    const char* name = luaL_checkstring(L, 1);
    unsigned int mask = 0;

    if (strcmp(name, "codes_ric") == 0 || strcmp(name, "CODES_RIC") == 0) {
        mask = AM_PACK_CODES_RIC;
    } else if (strcmp(name, "darkmatter") == 0 || strcmp(name, "DARKMATTER") == 0) {
        mask = AM_PACK_DARKMATTER;
    } else if (strcmp(name, "notorch") == 0 || strcmp(name, "NOTORCH") == 0) {
        mask = AM_PACK_NOTORCH;
    }

    lua_pushboolean(L, am_pack_enabled(mask));
    return 1;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUA -> C: COMMAND FUNCTIONS (modify AMK state)
// ═══════════════════════════════════════════════════════════════════════════════

static int l_amk_set_prophecy(lua_State* L) {
    int n = (int)luaL_checkinteger(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "PROPHECY %d", n);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_destiny(lua_State* L) {
    float f = (float)luaL_checknumber(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "DESTINY %.4f", f);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_wormhole(lua_State* L) {
    float f = (float)luaL_checknumber(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "WORMHOLE %.4f", f);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_pain(lua_State* L) {
    float f = (float)luaL_checknumber(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "PAIN %.4f", f);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_tension(lua_State* L) {
    float f = (float)luaL_checknumber(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "TENSION %.4f", f);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_dissonance(lua_State* L) {
    float f = (float)luaL_checknumber(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "DISSONANCE %.4f", f);
    am_exec(cmd);
    return 0;
}

static int l_amk_set_velocity(lua_State* L) {
    const char* mode = luaL_checkstring(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "VELOCITY %s", mode);
    am_exec(cmd);
    return 0;
}

static int l_amk_jump(lua_State* L) {
    int n = (int)luaL_checkinteger(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "JUMP %d", n);
    am_exec(cmd);
    return 0;
}

static int l_amk_exec(lua_State* L) {
    const char* script = luaL_checkstring(L, 1);
    int result = am_exec(script);
    lua_pushinteger(L, result);
    return 1;
}

static int l_amk_reset_field(lua_State* L) {
    (void)L;
    am_reset_field();
    return 0;
}

static int l_amk_reset_debt(lua_State* L) {
    (void)L;
    am_reset_debt();
    return 0;
}

static int l_amk_enable_pack(lua_State* L) {
    const char* name = luaL_checkstring(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "MODE %s", name);
    am_exec(cmd);
    return 0;
}

static int l_amk_disable_pack(lua_State* L) {
    const char* name = luaL_checkstring(L, 1);
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "DISABLE %s", name);
    am_exec(cmd);
    return 0;
}

static int l_amk_step(lua_State* L) {
    float dt = (float)luaL_checknumber(L, 1);
    am_step(dt);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LUA FUNCTION TABLE
// ═══════════════════════════════════════════════════════════════════════════════

static const luaL_Reg amk_funcs[] = {
    // Query functions
    {"prophecy", l_amk_prophecy},
    {"destiny", l_amk_destiny},
    {"wormhole", l_amk_wormhole},
    {"pain", l_amk_pain},
    {"tension", l_amk_tension},
    {"dissonance", l_amk_dissonance},
    {"velocity", l_amk_velocity},
    {"effective_temp", l_amk_effective_temp},
    {"debt", l_amk_debt},
    {"temporal_debt", l_amk_temporal_debt},
    {"pack_enabled", l_amk_pack_enabled},

    // Command functions
    {"set_prophecy", l_amk_set_prophecy},
    {"set_destiny", l_amk_set_destiny},
    {"set_wormhole", l_amk_set_wormhole},
    {"set_pain", l_amk_set_pain},
    {"set_tension", l_amk_set_tension},
    {"set_dissonance", l_amk_set_dissonance},
    {"set_velocity", l_amk_set_velocity},
    {"jump", l_amk_jump},
    {"exec", l_amk_exec},
    {"reset_field", l_amk_reset_field},
    {"reset_debt", l_amk_reset_debt},
    {"enable_pack", l_amk_enable_pack},
    {"disable_pack", l_amk_disable_pack},
    {"step", l_amk_step},

    {NULL, NULL}
};

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

int amk_lua_init(void) {
    if (g_initialized) return 0;

    L = luaL_newstate();
    if (!L) {
        fprintf(stderr, "[AMK Lua] Failed to create Lua state\n");
        return -1;
    }

    luaL_openlibs(L);

    // Create 'amk' table
    luaL_newlib(L, amk_funcs);
    lua_setglobal(L, "amk");

    // Add velocity constants
    lua_getglobal(L, "amk");
    lua_pushinteger(L, AM_VEL_NOMOVE);
    lua_setfield(L, -2, "VEL_NOMOVE");
    lua_pushinteger(L, AM_VEL_WALK);
    lua_setfield(L, -2, "VEL_WALK");
    lua_pushinteger(L, AM_VEL_RUN);
    lua_setfield(L, -2, "VEL_RUN");
    lua_pushinteger(L, AM_VEL_BACKWARD);
    lua_setfield(L, -2, "VEL_BACKWARD");
    lua_pop(L, 1);

    g_initialized = 1;
    printf("[AMK Lua] Initialized (Lua %s)\n", LUA_VERSION);

    return 0;
}

void amk_lua_shutdown(void) {
    if (L) {
        lua_close(L);
        L = NULL;
    }
    g_initialized = 0;
    g_last_script_path[0] = '\0';
}

lua_State* amk_lua_state(void) {
    return L;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCRIPT EXECUTION
// ═══════════════════════════════════════════════════════════════════════════════

int amk_lua_load(const char* path) {
    if (!L) {
        fprintf(stderr, "[AMK Lua] Not initialized\n");
        return -1;
    }

    // Save path for reload
    strncpy(g_last_script_path, path, sizeof(g_last_script_path) - 1);
    g_last_script_path[sizeof(g_last_script_path) - 1] = '\0';

    if (luaL_loadfile(L, path) != LUA_OK) {
        lua_error_handler(L, "load");
        return -1;
    }

    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        lua_error_handler(L, "run");
        return -1;
    }

    printf("[AMK Lua] Loaded: %s\n", path);
    return 0;
}

int amk_lua_exec(const char* code) {
    if (!L) {
        fprintf(stderr, "[AMK Lua] Not initialized\n");
        return -1;
    }

    if (luaL_loadstring(L, code) != LUA_OK) {
        lua_error_handler(L, "parse");
        return -1;
    }

    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        lua_error_handler(L, "exec");
        return -1;
    }

    return 0;
}

int amk_lua_reload(void) {
    if (g_last_script_path[0] == '\0') {
        fprintf(stderr, "[AMK Lua] No script loaded to reload\n");
        return -1;
    }

    printf("[AMK Lua] Reloading: %s\n", g_last_script_path);
    return amk_lua_load(g_last_script_path);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CALLBACKS (called from C at specific points)
// ═══════════════════════════════════════════════════════════════════════════════

void amk_lua_on_generate_start(const char* prompt, int max_tokens, float temperature) {
    if (!L) return;

    lua_getglobal(L, "on_generate_start");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    lua_pushstring(L, prompt);
    lua_pushinteger(L, max_tokens);
    lua_pushnumber(L, temperature);

    if (lua_pcall(L, 3, 0, 0) != LUA_OK) {
        lua_error_handler(L, "on_generate_start");
    }
}

void amk_lua_on_generate_end(const char* output, int tokens_generated) {
    if (!L) return;

    lua_getglobal(L, "on_generate_end");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    lua_pushstring(L, output);
    lua_pushinteger(L, tokens_generated);

    if (lua_pcall(L, 2, 0, 0) != LUA_OK) {
        lua_error_handler(L, "on_generate_end");
    }
}

void amk_lua_on_token(int token_id, float prob, int position) {
    if (!L) return;

    lua_getglobal(L, "on_token");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    lua_pushinteger(L, token_id);
    lua_pushnumber(L, prob);
    lua_pushinteger(L, position);

    if (lua_pcall(L, 3, 0, 0) != LUA_OK) {
        lua_error_handler(L, "on_token");
    }
}

void amk_lua_on_trauma(float intensity, const char* trigger) {
    if (!L) return;

    lua_getglobal(L, "on_trauma");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    lua_pushnumber(L, intensity);
    lua_pushstring(L, trigger ? trigger : "");

    if (lua_pcall(L, 2, 0, 0) != LUA_OK) {
        lua_error_handler(L, "on_trauma");
    }
}

void amk_lua_on_emotion(float valence, float arousal) {
    if (!L) return;

    lua_getglobal(L, "on_emotion");
    if (!lua_isfunction(L, -1)) {
        lua_pop(L, 1);
        return;
    }

    lua_pushnumber(L, valence);
    lua_pushnumber(L, arousal);

    if (lua_pcall(L, 2, 0, 0) != LUA_OK) {
        lua_error_handler(L, "on_emotion");
    }
}

#endif /* USE_LUA */
