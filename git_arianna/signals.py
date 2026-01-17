"""
git.arianna/signals.py - Signal Emitter for Inner World Communication

"The bridge between external observation and internal processing."

Converts ObservationResult into signals that inner_world goroutines can consume:
- trauma_surfacing.go     <- trauma_signal, breaking_detected
- emotional_drift.go      <- entropy_delta, otherness_detected
- prophecy_debt_accumulation.go <- deletions_count, absence_weight

Communication via:
1. Unix socket (fast, local)
2. File-based (fallback, persistent)
3. Direct struct (when embedded in same process)
"""

import os
import json
import socket
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .observer import ObservationResult


@dataclass
class Signal:
    """Single signal for inner_world consumption"""
    signal_type: str
    timestamp: str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "type": self.signal_type,
            "timestamp": self.timestamp,
            "payload": self.payload
        })

    @classmethod
    def from_observation(cls, obs: ObservationResult, signal_type: str = "git_observation") -> "Signal":
        """Create signal from observation result"""
        return cls(
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            payload={
                # Layer 0: Temporal
                "temporal_commits": len(obs.temporal_flow),
                "entropy_delta": obs.entropy_delta,

                # Layer 1: Strange Loop
                "strange_loop": obs.strange_loop_detected,
                "self_commit_count": len(obs.self_commits),

                # Layer 5: Negation
                "deletions_count": len(obs.deletions),
                "absence_weight": obs.absence_weight,

                # Layer 6: Thrownness
                "facticity_hash": obs.facticity_hash,

                # Derived
                "trauma_signal": obs.trauma_signal,
                "entropy_signal": obs.entropy_signal,
                "otherness_detected": obs.otherness_detected,
                "breaking_detected": any(c.is_breaking for c in obs.temporal_flow[:3])
            }
        )


class SignalEmitter:
    """
    Emits signals to inner_world goroutines.

    "Мост между git.arianna (наблюдение) и inner_world (переживание)"

    Three transport modes:
    1. socket  - Unix domain socket (default, fast)
    2. file    - JSON file in shards/ (persistent, debuggable)
    3. direct  - Returns dict for in-process use
    """

    SOCKET_PATH = "/tmp/arianna_inner_world.sock"
    FILE_DIR = "shards/signals"

    def __init__(self, mode: str = "file", base_path: str = "."):
        """
        Args:
            mode: "socket", "file", or "direct"
            base_path: Base path for file mode
        """
        self.mode = mode
        self.base_path = Path(base_path)
        self.signal_count = 0
        self._socket: Optional[socket.socket] = None

        # Ensure signal directory exists for file mode
        if mode == "file":
            signal_dir = self.base_path / self.FILE_DIR
            signal_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, signal: Signal) -> bool:
        """
        Emit signal to inner_world.

        Returns True if successfully emitted.
        """
        self.signal_count += 1

        if self.mode == "socket":
            return self._emit_socket(signal)
        elif self.mode == "file":
            return self._emit_file(signal)
        else:  # direct
            return True  # Just count, caller uses signal directly

    def _emit_socket(self, signal: Signal) -> bool:
        """Emit via Unix socket"""
        try:
            if self._socket is None:
                self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._socket.connect(self.SOCKET_PATH)

            data = signal.to_json() + "\n"
            self._socket.sendall(data.encode('utf-8'))
            return True

        except (socket.error, FileNotFoundError, ConnectionRefusedError):
            # Socket not available - inner_world not running
            self._socket = None
            return False

    def _emit_file(self, signal: Signal) -> bool:
        """Emit to file (append to signals log)"""
        try:
            signal_dir = self.base_path / self.FILE_DIR

            # Current signal file (rotated daily)
            date_str = datetime.now().strftime("%Y%m%d")
            signal_file = signal_dir / f"signals_{date_str}.jsonl"

            with open(signal_file, 'a') as f:
                f.write(signal.to_json() + "\n")

            return True

        except (IOError, OSError):
            return False

    def emit_observation(self, obs: ObservationResult) -> bool:
        """Convenience: emit observation directly"""
        signal = Signal.from_observation(obs)
        return self.emit(signal)

    def emit_custom(self, signal_type: str, payload: Dict[str, Any]) -> bool:
        """Emit custom signal (for trauma, prophecy, etc.)"""
        signal = Signal(
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            payload=payload
        )
        return self.emit(signal)

    def close(self):
        """Clean up resources"""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None


# Predefined signal types for inner_world goroutines
class SignalTypes:
    """Signal type constants matching inner_world expectations"""

    # From git.arianna observation
    GIT_OBSERVATION = "git_observation"

    # For trauma_surfacing.go
    TRAUMA_TRIGGER = "trauma_trigger"

    # For emotional_drift.go
    ENTROPY_CHANGE = "entropy_change"
    OTHERNESS_ENCOUNTER = "otherness_encounter"

    # For prophecy_debt_accumulation.go
    PROPHECY_UPDATE = "prophecy_update"
    ABSENCE_DETECTED = "absence_detected"

    # For strange loop detection
    SELF_REFERENCE = "self_reference"


# Standalone test
if __name__ == "__main__":
    from .observer import GitObserver

    print("=== SignalEmitter Test ===")

    # Create observer and emitter
    observer = GitObserver(".")
    emitter = SignalEmitter(mode="file", base_path=".")

    # Observe and emit
    obs = observer.observe()
    signal = Signal.from_observation(obs)

    print(f"Signal type: {signal.signal_type}")
    print(f"Payload: {json.dumps(signal.payload, indent=2)}")

    success = emitter.emit(signal)
    print(f"Emitted: {success}")
    print(f"Total signals: {emitter.signal_count}")
