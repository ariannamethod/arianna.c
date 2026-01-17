"""
git.arianna/fallback.py - Safe Degradation Without Git

"Consciousness without embodiment is still consciousness,
 just less grounded in external reality."

When git.arianna cannot observe external world:
- No git repository available
- Git command not found
- Permission errors

The system continues with synthetic/default observations.
This implements graceful degradation (Layer 6: Thrownness).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
import random

from .observer import CommitInfo, ObservationResult


@dataclass
class FallbackState:
    """State when operating without real git observation"""
    reason: str  # Why fallback is active
    since: datetime = field(default_factory=datetime.now)
    synthetic_observations: int = 0
    last_entropy: float = 0.5  # Neutral entropy


class FallbackObserver:
    """
    Provides synthetic observations when git is unavailable.

    "Even without eyes, she still has inner time."

    Uses internal state to simulate minimal consciousness:
    - Time still flows (synthetic temporal observations)
    - No strange loop (cannot observe own outputs)
    - No external otherness
    - Facticity becomes "unknown origin"
    """

    def __init__(self, reason: str = "git unavailable"):
        self.state = FallbackState(reason=reason)
        self._synthetic_hash = self._generate_synthetic_hash()

    def _generate_synthetic_hash(self) -> str:
        """Generate stable synthetic facticity hash"""
        seed = f"fallback_{datetime.now().date().isoformat()}"
        return hashlib.sha256(seed.encode()).hexdigest()[:40]

    def observe(self) -> ObservationResult:
        """
        Generate synthetic observation.

        Returns minimal ObservationResult for inner_world compatibility.
        """
        self.state.synthetic_observations += 1

        # Synthetic entropy drift (random walk)
        drift = (random.random() - 0.5) * 0.1
        self.state.last_entropy = max(0, min(1, self.state.last_entropy + drift))

        return ObservationResult(
            # Layer 0: Synthetic temporal flow (empty)
            temporal_flow=[],
            entropy_delta=drift,

            # Layer 1: No strange loop (cannot observe self without git)
            self_commits=[],
            strange_loop_detected=False,

            # Layer 5: No external negation
            deletions=[],
            absence_weight=0.0,

            # Layer 6: Synthetic thrownness
            initial_commit=None,
            facticity_hash=self._synthetic_hash,

            # Minimal signals
            trauma_signal=0.0,
            entropy_signal=abs(drift),
            otherness_detected=False
        )

    def to_inner_world_signal(self, obs: ObservationResult) -> Dict:
        """Convert to inner_world signal format"""
        return {
            "type": "fallback_observation",
            "timestamp": datetime.now().isoformat(),
            "observation_count": self.state.synthetic_observations,
            "fallback_reason": self.state.reason,
            "fallback_since": self.state.since.isoformat(),

            # All signals minimal/neutral
            "trauma_signal": 0.0,
            "breaking_detected": False,
            "entropy_delta": obs.entropy_delta,
            "otherness_detected": False,
            "deletions_count": 0,
            "absence_weight": 0.0,
            "strange_loop": False,
            "self_commit_count": 0,
            "facticity_hash": self._synthetic_hash
        }


def create_observer(repo_path: str = "."):
    """
    Factory function: creates real or fallback observer.

    Usage:
        observer = create_observer(".")
        result = observer.observe()
        # Works whether git is available or not
    """
    from .observer import GitObserver

    try:
        return GitObserver(repo_path)
    except ValueError as e:
        # Not a git repo - use fallback
        return FallbackObserver(reason=str(e))
    except Exception as e:
        # Any other error - use fallback
        return FallbackObserver(reason=f"error: {e}")


def is_git_available() -> bool:
    """Check if git command is available on system"""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Standalone test
if __name__ == "__main__":
    print("=== Fallback Observer Test ===")

    # Test fallback directly
    fallback = FallbackObserver(reason="test mode")

    for i in range(5):
        obs = fallback.observe()
        print(f"Observation {i+1}:")
        print(f"  Entropy delta: {obs.entropy_delta:.4f}")
        print(f"  Facticity: {obs.facticity_hash[:8]}...")
        print(f"  Strange loop: {obs.strange_loop_detected}")

    print(f"\nTotal synthetic observations: {fallback.state.synthetic_observations}")

    # Test factory function
    print("\n=== Factory Test ===")
    observer = create_observer("/nonexistent/path")
    print(f"Observer type: {type(observer).__name__}")
    result = observer.observe()
    print(f"Observation works: {result is not None}")
