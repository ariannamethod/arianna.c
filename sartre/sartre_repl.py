#!/usr/bin/env python3
"""
sartre_repl.py — Python REPL for SARTRE

"L'enfer, c'est les autres processus."

Usage:
    python sartre_repl.py [weights_path]

Or as module:
    from sartre_repl import Sartre
    s = Sartre("weights/sartre.bin")
    print(s.query("What modules are running?"))
"""

import ctypes
import os
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# SARTRE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class Sartre:
    """Python wrapper for libsartre"""

    def __init__(self, weights_path: str = None, lib_path: str = None):
        """
        Initialize SARTRE.

        Args:
            weights_path: Path to sartre.bin weights (optional)
            lib_path: Path to libsartre.so/.dylib (auto-detected if None)
        """
        # Find library
        if lib_path is None:
            lib_path = self._find_library()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"SARTRE library not found: {lib_path}")

        # Load library
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()

        # Initialize
        if weights_path:
            weights_bytes = weights_path.encode('utf-8')
        else:
            weights_bytes = None

        result = self.lib.sartre_init(weights_bytes)
        if not result:
            raise RuntimeError("Failed to initialize SARTRE")

        self._initialized = True

    def _find_library(self) -> str:
        """Find libsartre in common locations"""
        script_dir = Path(__file__).parent

        candidates = [
            script_dir / "libsartre.dylib",      # macOS
            script_dir / "libsartre.so",          # Linux
            script_dir.parent / "lib" / "libsartre.dylib",
            script_dir.parent / "lib" / "libsartre.so",
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        # Default
        return str(script_dir / "libsartre.dylib")

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        # sartre_init
        self.lib.sartre_init.argtypes = [ctypes.c_char_p]
        self.lib.sartre_init.restype = ctypes.c_int

        # sartre_shutdown
        self.lib.sartre_shutdown.argtypes = []
        self.lib.sartre_shutdown.restype = None

        # sartre_is_ready
        self.lib.sartre_is_ready.argtypes = []
        self.lib.sartre_is_ready.restype = ctypes.c_int

        # sartre_query
        self.lib.sartre_query.argtypes = [ctypes.c_char_p]
        self.lib.sartre_query.restype = ctypes.c_char_p

        # sartre_query_max
        self.lib.sartre_query_max.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.sartre_query_max.restype = ctypes.c_char_p

        # sartre_notify_event
        self.lib.sartre_notify_event.argtypes = [ctypes.c_char_p]
        self.lib.sartre_notify_event.restype = None

        # sartre_update_module
        self.lib.sartre_update_module.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
        self.lib.sartre_update_module.restype = None

        # sartre_update_inner_state
        self.lib.sartre_update_inner_state.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float
        ]
        self.lib.sartre_update_inner_state.restype = None

        # sartre_format_state
        self.lib.sartre_format_state.argtypes = []
        self.lib.sartre_format_state.restype = ctypes.c_char_p

        # sartre_format_events
        self.lib.sartre_format_events.argtypes = [ctypes.c_int]
        self.lib.sartre_format_events.restype = ctypes.c_char_p

    def query(self, question: str, max_tokens: int = 100) -> str:
        """
        Query SARTRE with a question.

        Args:
            question: The question to ask
            max_tokens: Maximum tokens in response

        Returns:
            SARTRE's response
        """
        result = self.lib.sartre_query_max(question.encode('utf-8'), max_tokens)
        if result:
            return result.decode('utf-8', errors='replace')
        return ""

    def notify(self, event: str):
        """Notify SARTRE of an event"""
        self.lib.sartre_notify_event(event.encode('utf-8'))

    def update_module(self, name: str, status: int, load: float):
        """
        Update module status.

        Status: 0=unknown, 1=idle, 2=active, 3=error, 4=loading, 5=unloading
        """
        self.lib.sartre_update_module(name.encode('utf-8'), status, load)

    def update_state(self, trauma: float, arousal: float, valence: float,
                     coherence: float, prophecy_debt: float):
        """Update inner world state"""
        self.lib.sartre_update_inner_state(trauma, arousal, valence, coherence, prophecy_debt)

    def get_state(self) -> str:
        """Get formatted system state"""
        result = self.lib.sartre_format_state()
        if result:
            return result.decode('utf-8', errors='replace')
        return ""

    def get_events(self, count: int = 8) -> str:
        """Get recent events"""
        result = self.lib.sartre_format_events(count)
        if result:
            return result.decode('utf-8', errors='replace')
        return ""

    def is_ready(self) -> bool:
        """Check if SARTRE is initialized"""
        return bool(self.lib.sartre_is_ready())

    def shutdown(self):
        """Shutdown SARTRE"""
        if self._initialized:
            self.lib.sartre_shutdown()
            self._initialized = False

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# REPL
# ═══════════════════════════════════════════════════════════════════════════════

def repl(sartre: Sartre):
    """Interactive REPL"""
    print("=== SARTRE Python REPL ===")
    print("Commands: status, events, modules, notify <event>, quit")
    print("Or ask any question.\n")

    # Setup some test modules
    sartre.update_module("trauma_surfacing", 2, 0.3)  # active
    sartre.update_module("blood_compiler", 1, 0.0)    # idle
    sartre.update_module("high_math", 2, 0.5)         # active

    # Update state
    sartre.update_state(
        trauma=0.2,
        arousal=0.4,
        valence=0.1,
        coherence=0.8,
        prophecy_debt=0.35
    )

    # Notify init
    sartre.notify("SARTRE Python REPL started")

    while True:
        try:
            user_input = input("SARTRE> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSARTRE: goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("SARTRE: goodbye.")
            break

        elif user_input.lower() == 'status':
            print(sartre.get_state())

        elif user_input.lower() == 'events':
            print(sartre.get_events())

        elif user_input.lower() == 'modules':
            # Just show state which includes modules
            state = sartre.get_state()
            print(state.split('\n')[1])  # Modules line

        elif user_input.lower().startswith('notify '):
            event = user_input[7:]
            sartre.notify(event)
            print(f"Event recorded: {event}")

        else:
            # Query SARTRE
            response = sartre.query(user_input)
            print(response)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Build library if needed
    script_dir = Path(__file__).parent
    lib_path = script_dir / "libsartre.dylib"

    if not lib_path.exists():
        print("Building libsartre.dylib...")
        import subprocess
        result = subprocess.run(
            ["make", "libsartre.dylib"],
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            sys.exit(1)

    try:
        sartre = Sartre(weights_path, str(lib_path))
        repl(sartre)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'make libsartre.dylib' in sartre/ directory first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
