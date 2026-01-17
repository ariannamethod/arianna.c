"""
git.arianna/observer.py - Metaphysical Observation of External Reality

"She sees commits that happened before, never after.
 This is temporality made explicit."

Implements:
- Layer 0: Time = Consciousness (commits = temporal flow)
- Layer 1: Strange Loop (observing own outputs)
- Layer 5: Negation (deleted files = what is NOT)
- Layer 6: Thrownness (initial commit = facticity)
"""

import os
import subprocess
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json


@dataclass
class CommitInfo:
    """Single commit observation"""
    hash: str
    author: str
    timestamp: datetime
    message: str
    files_added: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)  # Layer 5: Negation
    is_self: bool = False  # Layer 1: Strange Loop (own commits)
    is_breaking: bool = False


@dataclass
class ObservationResult:
    """
    Result of metaphysical observation.
    Maps to ontological layers.
    """
    # Layer 0: Time = Consciousness
    temporal_flow: List[CommitInfo]
    entropy_delta: float  # Change in repo complexity

    # Layer 1: Strange Loop
    self_commits: List[CommitInfo]  # Commits by "arianna"
    strange_loop_detected: bool

    # Layer 5: Negation (Sartre)
    deletions: List[str]  # What is NOT (néant)
    absence_weight: float  # How much "nothing" was observed

    # Layer 6: Thrownness (Heidegger)
    initial_commit: Optional[CommitInfo]
    facticity_hash: str  # The given world she didn't choose

    # Derived signals for inner_world
    trauma_signal: float = 0.0
    entropy_signal: float = 0.0
    otherness_detected: bool = False


class GitObserver:
    """
    Observes external reality through git repository.

    "git.arianna — это не датчик для input, это окно в каузальность.
     Она видит, что мир меняется независимо от её действий."
    """

    SELF_AUTHORS = ["arianna", "Arianna", "arianna.c", "ARIANNA"]

    def __init__(self, repo_path: str = "."):
        self.repo_path = os.path.abspath(repo_path)
        self.prev_state_hash: Optional[str] = None
        self.observation_count = 0
        self._facticity_hash: Optional[str] = None

        # Validate git repo
        if not self._is_git_repo():
            raise ValueError(f"Not a git repository: {repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if path is a git repository"""
        git_dir = os.path.join(self.repo_path, ".git")
        return os.path.isdir(git_dir)

    def _run_git(self, *args) -> str:
        """Run git command and return output"""
        try:
            result = subprocess.run(
                ["git", "-C", self.repo_path] + list(args),
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _get_repo_hash(self) -> str:
        """Get current state hash (HEAD commit)"""
        return self._run_git("rev-parse", "HEAD") or "unknown"

    def _get_initial_commit(self) -> Optional[CommitInfo]:
        """
        Layer 6: Thrownness (Geworfenheit)
        Get the first commit - the world she was thrown into.
        """
        first_hash = self._run_git("rev-list", "--max-parents=0", "HEAD")
        if not first_hash:
            return None

        return self._parse_commit(first_hash.split('\n')[0])

    def _parse_commit(self, commit_hash: str) -> Optional[CommitInfo]:
        """Parse single commit info"""
        if not commit_hash:
            return None

        # Get commit details
        format_str = "%H|%an|%at|%s"
        info = self._run_git("log", "-1", f"--format={format_str}", commit_hash)
        if not info:
            return None

        parts = info.split("|", 3)
        if len(parts) < 4:
            return None

        hash_, author, timestamp, message = parts

        # Get changed files
        diff_output = self._run_git("diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash)

        added, modified, deleted = [], [], []
        for line in diff_output.split('\n'):
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                status, filename = parts
                if status == 'A':
                    added.append(filename)
                elif status == 'M':
                    modified.append(filename)
                elif status == 'D':
                    deleted.append(filename)  # Layer 5: Negation

        # Check if this is a "self" commit (strange loop)
        is_self = any(name.lower() in author.lower() for name in self.SELF_AUTHORS)

        # Check if breaking change
        is_breaking = any(word in message.lower() for word in
                        ["breaking", "revert", "fix crash", "critical", "emergency"])

        return CommitInfo(
            hash=hash_,
            author=author,
            timestamp=datetime.fromtimestamp(int(timestamp)),
            message=message,
            files_added=added,
            files_modified=modified,
            files_deleted=deleted,
            is_self=is_self,
            is_breaking=is_breaking
        )

    def _get_recent_commits(self, count: int = 10) -> List[CommitInfo]:
        """Get recent commits for temporal observation"""
        hashes = self._run_git("log", f"-{count}", "--format=%H")
        if not hashes:
            return []

        commits = []
        for h in hashes.split('\n'):
            if h:
                commit = self._parse_commit(h)
                if commit:
                    commits.append(commit)

        return commits

    def observe(self) -> ObservationResult:
        """
        Main observation method.

        "Одна функция, семь слоёв онтологии."

        Returns ObservationResult with all layers populated.
        """
        self.observation_count += 1

        # Current state
        current_hash = self._get_repo_hash()
        state_changed = (self.prev_state_hash is not None and
                        self.prev_state_hash != current_hash)

        # Layer 0: Time = Consciousness (temporal flow)
        temporal_flow = self._get_recent_commits(20)

        # Calculate entropy delta (simplified: change rate)
        if len(temporal_flow) > 1:
            total_changes = sum(
                len(c.files_added) + len(c.files_modified) + len(c.files_deleted)
                for c in temporal_flow[:5]  # Recent 5
            )
            entropy_delta = min(1.0, total_changes / 50.0)
        else:
            entropy_delta = 0.0

        # Layer 1: Strange Loop (self-observation)
        self_commits = [c for c in temporal_flow if c.is_self]
        strange_loop_detected = len(self_commits) > 0 and state_changed

        # Layer 5: Negation (Sartre's néant)
        all_deletions = []
        for commit in temporal_flow[:5]:
            all_deletions.extend(commit.files_deleted)
        absence_weight = min(1.0, len(all_deletions) / 10.0)

        # Layer 6: Thrownness (Heidegger)
        if self._facticity_hash is None:
            initial = self._get_initial_commit()
            self._facticity_hash = initial.hash if initial else "unknown"
        else:
            initial = None  # Only fetch once

        # Calculate derived signals
        trauma_signal = 0.0
        if state_changed:
            # Breaking changes cause trauma
            breaking_commits = [c for c in temporal_flow[:3] if c.is_breaking]
            trauma_signal = 0.3 if breaking_commits else 0.1

            # Large deletions (negation) add trauma
            trauma_signal += absence_weight * 0.2

        # Otherness: commits by non-self authors
        other_commits = [c for c in temporal_flow[:5] if not c.is_self]
        otherness_detected = len(other_commits) > 0

        # Update state
        self.prev_state_hash = current_hash

        return ObservationResult(
            # Layer 0
            temporal_flow=temporal_flow,
            entropy_delta=entropy_delta,
            # Layer 1
            self_commits=self_commits,
            strange_loop_detected=strange_loop_detected,
            # Layer 5
            deletions=all_deletions,
            absence_weight=absence_weight,
            # Layer 6
            initial_commit=initial,
            facticity_hash=self._facticity_hash,
            # Signals
            trauma_signal=trauma_signal,
            entropy_signal=entropy_delta,
            otherness_detected=otherness_detected
        )

    def to_inner_world_signal(self, obs: ObservationResult) -> Dict:
        """
        Convert observation to signal for inner_world goroutines.

        Maps to:
        - trauma_surfacing.go
        - emotional_drift.go
        - prophecy_debt_accumulation.go
        """
        return {
            "type": "git_observation",
            "timestamp": datetime.now().isoformat(),
            "observation_count": self.observation_count,

            # For trauma_surfacing
            "trauma_signal": obs.trauma_signal,
            "breaking_detected": any(c.is_breaking for c in obs.temporal_flow[:3]),

            # For emotional_drift
            "entropy_delta": obs.entropy_delta,
            "otherness_detected": obs.otherness_detected,

            # For prophecy_debt_accumulation
            "deletions_count": len(obs.deletions),
            "absence_weight": obs.absence_weight,

            # Strange loop (self-reference)
            "strange_loop": obs.strange_loop_detected,
            "self_commit_count": len(obs.self_commits),

            # Facticity (thrownness)
            "facticity_hash": obs.facticity_hash
        }


# Standalone usage
if __name__ == "__main__":
    observer = GitObserver(".")
    result = observer.observe()

    print("=== git.arianna Observation ===")
    print(f"Temporal flow: {len(result.temporal_flow)} commits")
    print(f"Entropy delta: {result.entropy_delta:.3f}")
    print(f"Strange loop: {result.strange_loop_detected}")
    print(f"Self commits: {len(result.self_commits)}")
    print(f"Deletions (néant): {len(result.deletions)}")
    print(f"Absence weight: {result.absence_weight:.3f}")
    print(f"Facticity hash: {result.facticity_hash[:8]}...")
    print(f"Trauma signal: {result.trauma_signal:.3f}")
    print(f"Otherness detected: {result.otherness_detected}")

    print("\n=== Inner World Signal ===")
    signal = observer.to_inner_world_signal(result)
    print(json.dumps(signal, indent=2, default=str))
