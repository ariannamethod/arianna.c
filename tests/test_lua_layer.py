#!/usr/bin/env python3
"""
tests/test_lua_layer.py — Tests for Lua scripting layer

Tests:
- Lua script syntax validation
- API completeness check
- Callback definitions
- Integration patterns

Note: These tests don't require Lua installed - they validate
the script structure and can mock the amk table.
"""

import unittest
import os
import re
from pathlib import Path


class TestLuaScriptSyntax(unittest.TestCase):
    """Test Lua script syntax and structure"""

    @classmethod
    def setUpClass(cls):
        cls.script_path = Path(__file__).parent.parent / "scripts" / "amk_default.lua"
        with open(cls.script_path, 'r') as f:
            cls.script_content = f.read()

    def test_script_exists(self):
        """Script file should exist"""
        self.assertTrue(self.script_path.exists())

    def test_script_not_empty(self):
        """Script should have content"""
        self.assertGreater(len(self.script_content), 100)

    def test_has_config_table(self):
        """Script should define config table"""
        self.assertIn("local config = {", self.script_content)

    def test_has_init_function(self):
        """Script should have init function"""
        self.assertIn("local function init()", self.script_content)

    def test_has_on_generate_start_callback(self):
        """Script should define on_generate_start callback"""
        self.assertIn("function on_generate_start(", self.script_content)

    def test_has_on_generate_end_callback(self):
        """Script should define on_generate_end callback"""
        self.assertIn("function on_generate_end(", self.script_content)

    def test_has_on_token_callback(self):
        """Script should define on_token callback"""
        self.assertIn("function on_token(", self.script_content)

    def test_has_on_trauma_callback(self):
        """Script should define on_trauma callback"""
        self.assertIn("function on_trauma(", self.script_content)

    def test_has_on_emotion_callback(self):
        """Script should define on_emotion callback"""
        self.assertIn("function on_emotion(", self.script_content)

    def test_has_apply_mood_function(self):
        """Script should have apply_mood utility"""
        self.assertIn("function apply_mood(", self.script_content)

    def test_has_mood_presets(self):
        """Script should define mood presets"""
        moods = ["calm", "intense", "contemplative", "chaotic", "nostalgic"]
        for mood in moods:
            self.assertIn(mood, self.script_content, f"Missing mood preset: {mood}")

    def test_calls_init(self):
        """Script should call init() at the end"""
        self.assertIn("init()", self.script_content)

    def test_no_syntax_errors_basic(self):
        """Basic syntax check - balanced brackets"""
        # Count opening and closing braces
        open_braces = self.script_content.count('{')
        close_braces = self.script_content.count('}')
        self.assertEqual(open_braces, close_braces, "Unbalanced braces")

        # Count function/end pairs (rough check)
        functions = len(re.findall(r'\bfunction\b', self.script_content))
        ends = len(re.findall(r'\bend\b', self.script_content))
        # Allow for 'end' in strings/comments
        self.assertGreaterEqual(ends, functions, "Possible missing 'end' keyword")


class TestLuaAPIUsage(unittest.TestCase):
    """Test that Lua script uses the expected API"""

    @classmethod
    def setUpClass(cls):
        cls.script_path = Path(__file__).parent.parent / "scripts" / "amk_default.lua"
        with open(cls.script_path, 'r') as f:
            cls.script_content = f.read()

    # Query functions
    def test_uses_amk_prophecy(self):
        """Script should use amk.prophecy()"""
        self.assertIn("amk.prophecy()", self.script_content)

    def test_uses_amk_destiny(self):
        """Script should use amk.destiny()"""
        self.assertIn("amk.destiny()", self.script_content)

    def test_uses_amk_wormhole(self):
        """Script should use amk.wormhole()"""
        self.assertIn("amk.wormhole()", self.script_content)

    def test_uses_amk_tension(self):
        """Script should use amk.tension()"""
        self.assertIn("amk.tension()", self.script_content)

    def test_uses_amk_pain(self):
        """Script should use amk.pain()"""
        self.assertIn("amk.pain()", self.script_content)

    def test_uses_amk_debt(self):
        """Script should use amk.debt()"""
        self.assertIn("amk.debt()", self.script_content)

    def test_uses_amk_velocity(self):
        """Script should use amk.velocity()"""
        self.assertIn("amk.velocity()", self.script_content)

    # Command functions
    def test_uses_set_prophecy(self):
        """Script should use amk.set_prophecy()"""
        self.assertIn("amk.set_prophecy(", self.script_content)

    def test_uses_set_destiny(self):
        """Script should use amk.set_destiny()"""
        self.assertIn("amk.set_destiny(", self.script_content)

    def test_uses_set_wormhole(self):
        """Script should use amk.set_wormhole()"""
        self.assertIn("amk.set_wormhole(", self.script_content)

    def test_uses_set_velocity(self):
        """Script should use amk.set_velocity()"""
        self.assertIn("amk.set_velocity(", self.script_content)

    def test_uses_set_pain(self):
        """Script should use amk.set_pain()"""
        self.assertIn("amk.set_pain(", self.script_content)

    def test_uses_set_tension(self):
        """Script should use amk.set_tension()"""
        self.assertIn("amk.set_tension(", self.script_content)

    def test_uses_amk_step(self):
        """Script should use amk.step()"""
        self.assertIn("amk.step(", self.script_content)

    # Constants
    def test_uses_vel_constants(self):
        """Script should use velocity constants"""
        self.assertIn("amk.VEL_", self.script_content)


class TestLuaCHeader(unittest.TestCase):
    """Test C header for Lua bindings"""

    @classmethod
    def setUpClass(cls):
        cls.header_path = Path(__file__).parent.parent / "src" / "amk_lua.h"
        with open(cls.header_path, 'r') as f:
            cls.header_content = f.read()

    def test_header_exists(self):
        """Header file should exist"""
        self.assertTrue(self.header_path.exists())

    def test_has_include_guard(self):
        """Header should have include guard"""
        self.assertIn("#ifndef AMK_LUA_H", self.header_content)
        self.assertIn("#define AMK_LUA_H", self.header_content)

    def test_has_use_lua_ifdef(self):
        """Header should have USE_LUA conditional"""
        self.assertIn("#ifdef USE_LUA", self.header_content)

    def test_has_init_function(self):
        """Header should declare amk_lua_init"""
        self.assertIn("int amk_lua_init(void)", self.header_content)

    def test_has_shutdown_function(self):
        """Header should declare amk_lua_shutdown"""
        self.assertIn("void amk_lua_shutdown(void)", self.header_content)

    def test_has_load_function(self):
        """Header should declare amk_lua_load"""
        self.assertIn("int amk_lua_load(const char* path)", self.header_content)

    def test_has_exec_function(self):
        """Header should declare amk_lua_exec"""
        self.assertIn("int amk_lua_exec(const char* code)", self.header_content)

    def test_has_reload_function(self):
        """Header should declare amk_lua_reload"""
        self.assertIn("int amk_lua_reload(void)", self.header_content)

    def test_has_callbacks(self):
        """Header should declare callback functions"""
        callbacks = [
            "amk_lua_on_generate_start",
            "amk_lua_on_generate_end",
            "amk_lua_on_token",
            "amk_lua_on_trauma",
            "amk_lua_on_emotion",
        ]
        for cb in callbacks:
            self.assertIn(cb, self.header_content, f"Missing callback: {cb}")

    def test_has_stubs(self):
        """Header should have stubs when USE_LUA not defined"""
        self.assertIn("static inline int amk_lua_init(void) { return 0; }", self.header_content)


class TestLuaCImplementation(unittest.TestCase):
    """Test C implementation for Lua bindings"""

    @classmethod
    def setUpClass(cls):
        cls.impl_path = Path(__file__).parent.parent / "src" / "amk_lua.c"
        with open(cls.impl_path, 'r') as f:
            cls.impl_content = f.read()

    def test_impl_exists(self):
        """Implementation file should exist"""
        self.assertTrue(self.impl_path.exists())

    def test_has_use_lua_ifdef(self):
        """Implementation should be wrapped in USE_LUA"""
        self.assertIn("#ifdef USE_LUA", self.impl_content)

    def test_includes_amk_kernel(self):
        """Implementation should include amk_kernel.h"""
        self.assertIn('#include "amk_kernel.h"', self.impl_content)

    def test_has_lua_state(self):
        """Implementation should have Lua state"""
        self.assertIn("lua_State* L", self.impl_content)

    def test_has_function_table(self):
        """Implementation should have function registration table"""
        self.assertIn("luaL_Reg amk_funcs[]", self.impl_content)

    def test_registers_query_functions(self):
        """Implementation should register query functions"""
        query_funcs = ["prophecy", "destiny", "wormhole", "pain", "tension"]
        for func in query_funcs:
            self.assertIn(f'"{func}"', self.impl_content, f"Missing query func: {func}")

    def test_registers_command_functions(self):
        """Implementation should register command functions"""
        cmd_funcs = ["set_prophecy", "set_destiny", "set_wormhole", "set_velocity"]
        for func in cmd_funcs:
            self.assertIn(f'"{func}"', self.impl_content, f"Missing command func: {func}")

    def test_creates_amk_table(self):
        """Implementation should create 'amk' global table"""
        self.assertIn('lua_setglobal(L, "amk")', self.impl_content)

    def test_has_velocity_constants(self):
        """Implementation should add velocity constants"""
        self.assertIn("VEL_NOMOVE", self.impl_content)
        self.assertIn("VEL_WALK", self.impl_content)
        self.assertIn("VEL_RUN", self.impl_content)
        self.assertIn("VEL_BACKWARD", self.impl_content)


class TestLuaDocumentation(unittest.TestCase):
    """Test that Lua layer is documented"""

    @classmethod
    def setUpClass(cls):
        cls.script_path = Path(__file__).parent.parent / "scripts" / "amk_default.lua"
        with open(cls.script_path, 'r') as f:
            cls.script_content = f.read()

    def test_has_header_comment(self):
        """Script should have header documentation"""
        self.assertTrue(self.script_content.startswith("--[["))

    def test_documents_query_api(self):
        """Script should document query API"""
        self.assertIn("amk.prophecy()", self.script_content[:2000])
        self.assertIn("amk.destiny()", self.script_content[:2000])

    def test_documents_command_api(self):
        """Script should document command API"""
        self.assertIn("amk.set_prophecy(", self.script_content[:2000])
        self.assertIn("amk.set_destiny(", self.script_content[:2000])

    def test_documents_callbacks(self):
        """Script should document callbacks"""
        self.assertIn("on_generate_start(", self.script_content[:2000])
        self.assertIn("on_trauma(", self.script_content[:2000])


if __name__ == "__main__":
    unittest.main(verbosity=2)
