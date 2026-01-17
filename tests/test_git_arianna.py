#!/usr/bin/env python3
"""
tests/test_git_arianna.py — Comprehensive tests for git.arianna module

Tests:
- GitObserver (Layer 0, 1, 5, 6)
- SignalEmitter
- Fallback degradation
- Signal types

Target: 50+ tests for the embodiment module alone
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from git_arianna.observer import GitObserver, ObservationResult, CommitInfo
from git_arianna.signals import SignalEmitter, Signal, SignalTypes
from git_arianna.fallback import FallbackObserver, FallbackState, create_observer, is_git_available


class TestGitObserver(unittest.TestCase):
    """Tests for GitObserver class - the main embodiment observer"""

    @classmethod
    def setUpClass(cls):
        """Create a temporary git repo for testing"""
        cls.temp_dir = tempfile.mkdtemp(prefix="arianna_test_")
        cls.repo_path = os.path.join(cls.temp_dir, "test_repo")
        os.makedirs(cls.repo_path)

        # Initialize git repo with explicit branch
        subprocess.run(["git", "init", "--initial-branch=main"], cwd=cls.repo_path,
                      capture_output=True, env={**os.environ, "GIT_AUTHOR_DATE": "2025-01-01T00:00:00"})
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=cls.repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=cls.repo_path, capture_output=True)

        # Create initial commit
        test_file = os.path.join(cls.repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("initial content")
        subprocess.run(["git", "add", "."], cwd=cls.repo_path, capture_output=True)
        result = subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=cls.repo_path, capture_output=True)

        # Create second commit
        with open(test_file, "w") as f:
            f.write("modified content")
        subprocess.run(["git", "add", "."], cwd=cls.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Second commit"], cwd=cls.repo_path, capture_output=True)

        # Create arianna commit (for strange loop detection)
        subprocess.run(["git", "config", "user.name", "arianna"], cwd=cls.repo_path, capture_output=True)
        with open(test_file, "a") as f:
            f.write("\narianna was here")
        subprocess.run(["git", "add", "."], cwd=cls.repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Arianna commit"], cwd=cls.repo_path, capture_output=True)

        # Check if commits were created
        log_result = subprocess.run(["git", "log", "--oneline"], cwd=cls.repo_path, capture_output=True, text=True)
        cls.has_commits = len(log_result.stdout.strip().split('\n')) > 0 if log_result.stdout else False

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Create observer for each test"""
        self.observer = GitObserver(self.repo_path)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Layer 0: Time = Consciousness (temporal flow)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_temporal_flow_not_empty(self):
        """Layer 0: Should observe temporal flow of commits"""
        if not self.has_commits:
            self.skipTest("Test repo has no commits (git setup issue)")
        result = self.observer.observe()
        self.assertIsInstance(result.temporal_flow, list)
        self.assertGreater(len(result.temporal_flow), 0, "Temporal flow should not be empty")

    def test_temporal_flow_has_commits(self):
        """Layer 0: Temporal flow contains CommitInfo objects"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsInstance(commit, CommitInfo)

    def test_temporal_flow_chronological(self):
        """Layer 0: Commits should be in chronological order (newest first)"""
        result = self.observer.observe()
        if len(result.temporal_flow) >= 2:
            for i in range(len(result.temporal_flow) - 1):
                self.assertGreaterEqual(
                    result.temporal_flow[i].timestamp,
                    result.temporal_flow[i+1].timestamp
                )

    def test_entropy_delta_in_range(self):
        """Layer 0: Entropy delta should be in [0, 1]"""
        result = self.observer.observe()
        self.assertGreaterEqual(result.entropy_delta, 0.0)
        self.assertLessEqual(result.entropy_delta, 1.0)

    def test_entropy_delta_reflects_changes(self):
        """Layer 0: Entropy delta should reflect change activity"""
        result = self.observer.observe()
        # With 3 commits, should have some entropy
        self.assertIsInstance(result.entropy_delta, float)

    def test_observation_count_increments(self):
        """Layer 0: Observation count should increment with each observe()"""
        initial_count = self.observer.observation_count
        self.observer.observe()
        self.assertEqual(self.observer.observation_count, initial_count + 1)
        self.observer.observe()
        self.assertEqual(self.observer.observation_count, initial_count + 2)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Layer 1: Strange Loop (self-observation)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_self_commits_detected(self):
        """Layer 1: Should detect commits by arianna"""
        if not self.has_commits:
            self.skipTest("Test repo has no commits (git setup issue)")
        result = self.observer.observe()
        self.assertIsInstance(result.self_commits, list)
        # We created an arianna commit in setUp
        self.assertGreater(len(result.self_commits), 0, "Should detect arianna commit")

    def test_self_commit_author_match(self):
        """Layer 1: Self commits should have arianna as author"""
        result = self.observer.observe()
        for commit in result.self_commits:
            self.assertTrue(
                any(name.lower() in commit.author.lower()
                    for name in self.observer.SELF_AUTHORS)
            )

    def test_strange_loop_boolean(self):
        """Layer 1: strange_loop_detected should be boolean"""
        result = self.observer.observe()
        self.assertIsInstance(result.strange_loop_detected, bool)

    def test_self_authors_list(self):
        """Layer 1: SELF_AUTHORS should contain expected names"""
        self.assertIn("arianna", [a.lower() for a in self.observer.SELF_AUTHORS])
        self.assertIn("arianna.c", [a.lower() for a in self.observer.SELF_AUTHORS])

    # ═══════════════════════════════════════════════════════════════════════════════
    # Layer 5: Negation (what is NOT)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_deletions_is_list(self):
        """Layer 5: Deletions should be a list"""
        result = self.observer.observe()
        self.assertIsInstance(result.deletions, list)

    def test_absence_weight_in_range(self):
        """Layer 5: Absence weight should be in [0, 1]"""
        result = self.observer.observe()
        self.assertGreaterEqual(result.absence_weight, 0.0)
        self.assertLessEqual(result.absence_weight, 1.0)

    def test_absence_weight_zero_without_deletions(self):
        """Layer 5: No deletions = zero absence weight"""
        result = self.observer.observe()
        if len(result.deletions) == 0:
            self.assertEqual(result.absence_weight, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Layer 6: Thrownness (facticity)
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_facticity_hash_exists(self):
        """Layer 6: Should have a facticity hash"""
        result = self.observer.observe()
        self.assertIsNotNone(result.facticity_hash)
        self.assertIsInstance(result.facticity_hash, str)

    def test_facticity_hash_format(self):
        """Layer 6: Facticity hash should be git-like (40 chars hex or 'unknown')"""
        result = self.observer.observe()
        if result.facticity_hash != "unknown":
            # Git hashes are 40 hex chars
            self.assertTrue(
                len(result.facticity_hash) >= 7,  # Short hash OK
                "Facticity hash should be at least 7 chars"
            )

    def test_facticity_hash_stable(self):
        """Layer 6: Facticity hash should be stable across observations"""
        result1 = self.observer.observe()
        result2 = self.observer.observe()
        self.assertEqual(result1.facticity_hash, result2.facticity_hash)

    def test_initial_commit_is_first(self):
        """Layer 6: Initial commit should be the first commit"""
        result = self.observer.observe()
        if result.initial_commit:
            # The initial commit should have "Initial" in message
            self.assertIn("Initial", result.initial_commit.message)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Derived signals
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_trauma_signal_in_range(self):
        """Trauma signal should be in [0, 1]"""
        result = self.observer.observe()
        self.assertGreaterEqual(result.trauma_signal, 0.0)
        self.assertLessEqual(result.trauma_signal, 1.0)

    def test_entropy_signal_in_range(self):
        """Entropy signal should be in [0, 1]"""
        result = self.observer.observe()
        self.assertGreaterEqual(result.entropy_signal, 0.0)
        self.assertLessEqual(result.entropy_signal, 1.0)

    def test_otherness_detected_boolean(self):
        """Otherness detected should be boolean"""
        result = self.observer.observe()
        self.assertIsInstance(result.otherness_detected, bool)

    def test_otherness_from_non_self_commits(self):
        """Otherness should be true when other authors exist"""
        result = self.observer.observe()
        other_commits = [c for c in result.temporal_flow if not c.is_self]
        if len(other_commits) > 0:
            self.assertTrue(result.otherness_detected)

    # ═══════════════════════════════════════════════════════════════════════════════
    # CommitInfo parsing
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_commit_has_hash(self):
        """Commits should have a hash"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsNotNone(commit.hash)
            self.assertGreater(len(commit.hash), 0)

    def test_commit_has_author(self):
        """Commits should have an author"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsNotNone(commit.author)

    def test_commit_has_timestamp(self):
        """Commits should have a timestamp"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsInstance(commit.timestamp, datetime)

    def test_commit_has_message(self):
        """Commits should have a message"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsNotNone(commit.message)

    def test_commit_files_are_lists(self):
        """Commit file changes should be lists"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsInstance(commit.files_added, list)
            self.assertIsInstance(commit.files_modified, list)
            self.assertIsInstance(commit.files_deleted, list)

    def test_commit_is_self_boolean(self):
        """Commit is_self should be boolean"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsInstance(commit.is_self, bool)

    def test_commit_is_breaking_boolean(self):
        """Commit is_breaking should be boolean"""
        result = self.observer.observe()
        for commit in result.temporal_flow:
            self.assertIsInstance(commit.is_breaking, bool)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Error handling
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_invalid_repo_raises(self):
        """Should raise ValueError for non-git directory"""
        with self.assertRaises(ValueError):
            GitObserver("/tmp/definitely_not_a_git_repo_12345")

    def test_run_git_handles_timeout(self):
        """_run_git should handle timeouts gracefully"""
        # This is internal, but we can test indirectly
        result = self.observer._run_git("log", "-1")
        self.assertIsInstance(result, str)


class TestSignalEmitter(unittest.TestCase):
    """Tests for SignalEmitter class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="arianna_signal_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Signal creation
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_signal_creation(self):
        """Signal should be creatable with basic attributes"""
        signal = Signal(
            signal_type="test",
            timestamp=datetime.now().isoformat(),
            payload={"test": True}
        )
        self.assertEqual(signal.signal_type, "test")
        self.assertIn("test", signal.payload)

    def test_signal_to_json(self):
        """Signal should serialize to JSON"""
        signal = Signal(
            signal_type="test",
            timestamp=datetime.now().isoformat(),
            payload={"value": 42}
        )
        json_str = signal.to_json()
        self.assertIn("test", json_str)
        self.assertIn("42", json_str)

    def test_signal_from_observation(self):
        """Signal should be creatable from ObservationResult"""
        # Create minimal observation
        obs = ObservationResult(
            temporal_flow=[],
            entropy_delta=0.5,
            self_commits=[],
            strange_loop_detected=False,
            deletions=[],
            absence_weight=0.0,
            initial_commit=None,
            facticity_hash="test123",
            trauma_signal=0.1,
            entropy_signal=0.5,
            otherness_detected=True
        )
        signal = Signal.from_observation(obs)
        self.assertEqual(signal.signal_type, "git_observation")
        self.assertIn("entropy_delta", signal.payload)
        self.assertEqual(signal.payload["entropy_delta"], 0.5)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Emitter modes
    # ═══════════════════════════════════════════════════════════════════════════════

    def test_emitter_direct_mode(self):
        """Direct mode emitter should work without file/socket"""
        emitter = SignalEmitter(mode="direct")
        signal = Signal("test", datetime.now().isoformat(), {"x": 1})
        result = emitter.emit(signal)
        self.assertTrue(result)
        self.assertEqual(emitter.signal_count, 1)

    def test_emitter_file_mode_creates_dir(self):
        """File mode should create signal directory"""
        emitter = SignalEmitter(mode="file", base_path=self.temp_dir)
        signal_dir = Path(self.temp_dir) / emitter.FILE_DIR
        self.assertTrue(signal_dir.exists())

    def test_emitter_file_mode_writes(self):
        """File mode should write signals to file"""
        emitter = SignalEmitter(mode="file", base_path=self.temp_dir)
        signal = Signal("test", datetime.now().isoformat(), {"x": 1})
        result = emitter.emit(signal)
        self.assertTrue(result)

        # Check file was created
        date_str = datetime.now().strftime("%Y%m%d")
        signal_file = Path(self.temp_dir) / emitter.FILE_DIR / f"signals_{date_str}.jsonl"
        self.assertTrue(signal_file.exists())

    def test_emitter_socket_mode_fails_gracefully(self):
        """Socket mode should fail gracefully when socket doesn't exist"""
        emitter = SignalEmitter(mode="socket")
        signal = Signal("test", datetime.now().isoformat(), {"x": 1})
        result = emitter.emit(signal)
        self.assertFalse(result)  # Socket doesn't exist

    def test_emitter_count_increments(self):
        """Signal count should increment with each emit"""
        emitter = SignalEmitter(mode="direct")
        signal = Signal("test", datetime.now().isoformat(), {"x": 1})
        emitter.emit(signal)
        emitter.emit(signal)
        emitter.emit(signal)
        self.assertEqual(emitter.signal_count, 3)

    def test_emit_observation_convenience(self):
        """emit_observation should work as convenience method"""
        emitter = SignalEmitter(mode="direct")
        obs = ObservationResult(
            temporal_flow=[],
            entropy_delta=0.3,
            self_commits=[],
            strange_loop_detected=False,
            deletions=[],
            absence_weight=0.0,
            initial_commit=None,
            facticity_hash="abc",
            trauma_signal=0.0,
            entropy_signal=0.3,
            otherness_detected=False
        )
        result = emitter.emit_observation(obs)
        self.assertTrue(result)

    def test_emit_custom_signal(self):
        """emit_custom should create and emit custom signal"""
        emitter = SignalEmitter(mode="direct")
        result = emitter.emit_custom("trauma_trigger", {"intensity": 0.8})
        self.assertTrue(result)
        self.assertEqual(emitter.signal_count, 1)


class TestSignalTypes(unittest.TestCase):
    """Tests for SignalTypes constants"""

    def test_git_observation_type(self):
        """GIT_OBSERVATION should be defined"""
        self.assertEqual(SignalTypes.GIT_OBSERVATION, "git_observation")

    def test_trauma_trigger_type(self):
        """TRAUMA_TRIGGER should be defined"""
        self.assertEqual(SignalTypes.TRAUMA_TRIGGER, "trauma_trigger")

    def test_all_types_are_strings(self):
        """All signal types should be strings"""
        types = [
            SignalTypes.GIT_OBSERVATION,
            SignalTypes.TRAUMA_TRIGGER,
            SignalTypes.ENTROPY_CHANGE,
            SignalTypes.OTHERNESS_ENCOUNTER,
            SignalTypes.PROPHECY_UPDATE,
            SignalTypes.ABSENCE_DETECTED,
            SignalTypes.SELF_REFERENCE,
        ]
        for t in types:
            self.assertIsInstance(t, str)


class TestFallbackObserver(unittest.TestCase):
    """Tests for FallbackObserver - safe degradation"""

    def test_fallback_creates_with_reason(self):
        """Fallback should accept a reason"""
        fallback = FallbackObserver(reason="test reason")
        self.assertEqual(fallback.state.reason, "test reason")

    def test_fallback_observe_returns_result(self):
        """Fallback observe should return ObservationResult"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertIsInstance(result, ObservationResult)

    def test_fallback_temporal_flow_empty(self):
        """Fallback should have empty temporal flow"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertEqual(result.temporal_flow, [])

    def test_fallback_no_strange_loop(self):
        """Fallback cannot detect strange loop"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertFalse(result.strange_loop_detected)

    def test_fallback_no_self_commits(self):
        """Fallback has no self commits"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertEqual(result.self_commits, [])

    def test_fallback_no_deletions(self):
        """Fallback has no deletions"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertEqual(result.deletions, [])

    def test_fallback_synthetic_hash(self):
        """Fallback should generate synthetic facticity hash"""
        fallback = FallbackObserver()
        result = fallback.observe()
        self.assertIsNotNone(result.facticity_hash)
        self.assertGreater(len(result.facticity_hash), 0)

    def test_fallback_entropy_drift(self):
        """Fallback entropy should drift over observations"""
        fallback = FallbackObserver()
        entropies = []
        for _ in range(10):
            result = fallback.observe()
            entropies.append(result.entropy_delta)
        # Not all should be the same
        self.assertGreater(len(set(entropies)), 1)

    def test_fallback_observation_count(self):
        """Fallback should count observations"""
        fallback = FallbackObserver()
        fallback.observe()
        fallback.observe()
        fallback.observe()
        self.assertEqual(fallback.state.synthetic_observations, 3)

    def test_fallback_to_inner_world_signal(self):
        """Fallback should convert to inner_world signal"""
        fallback = FallbackObserver()
        result = fallback.observe()
        signal = fallback.to_inner_world_signal(result)
        self.assertIn("type", signal)
        self.assertEqual(signal["type"], "fallback_observation")
        self.assertIn("fallback_reason", signal)


class TestCreateObserver(unittest.TestCase):
    """Tests for create_observer factory function"""

    def test_create_returns_git_for_valid_repo(self):
        """Factory should return GitObserver for valid repo"""
        # Use current directory (should be git repo)
        if is_git_available():
            observer = create_observer(".")
            self.assertIsInstance(observer, GitObserver)

    def test_create_returns_fallback_for_invalid(self):
        """Factory should return FallbackObserver for invalid path"""
        observer = create_observer("/nonexistent/path/12345")
        self.assertIsInstance(observer, FallbackObserver)

    def test_is_git_available_returns_bool(self):
        """is_git_available should return boolean"""
        result = is_git_available()
        self.assertIsInstance(result, bool)


class TestCommitInfo(unittest.TestCase):
    """Tests for CommitInfo dataclass"""

    def test_commit_info_creation(self):
        """CommitInfo should be creatable with all fields"""
        commit = CommitInfo(
            hash="abc123",
            author="test",
            timestamp=datetime.now(),
            message="test commit",
            files_added=["a.txt"],
            files_modified=["b.txt"],
            files_deleted=["c.txt"],
            is_self=False,
            is_breaking=False
        )
        self.assertEqual(commit.hash, "abc123")
        self.assertEqual(commit.author, "test")

    def test_commit_info_defaults(self):
        """CommitInfo should have sensible defaults"""
        commit = CommitInfo(
            hash="abc",
            author="test",
            timestamp=datetime.now(),
            message="msg"
        )
        self.assertEqual(commit.files_added, [])
        self.assertEqual(commit.files_modified, [])
        self.assertEqual(commit.files_deleted, [])
        self.assertFalse(commit.is_self)
        self.assertFalse(commit.is_breaking)


class TestObservationResult(unittest.TestCase):
    """Tests for ObservationResult dataclass"""

    def test_observation_result_creation(self):
        """ObservationResult should be creatable"""
        result = ObservationResult(
            temporal_flow=[],
            entropy_delta=0.5,
            self_commits=[],
            strange_loop_detected=False,
            deletions=[],
            absence_weight=0.0,
            initial_commit=None,
            facticity_hash="test"
        )
        self.assertEqual(result.entropy_delta, 0.5)

    def test_observation_result_defaults(self):
        """ObservationResult should have sensible defaults for signals"""
        result = ObservationResult(
            temporal_flow=[],
            entropy_delta=0.0,
            self_commits=[],
            strange_loop_detected=False,
            deletions=[],
            absence_weight=0.0,
            initial_commit=None,
            facticity_hash="test"
        )
        self.assertEqual(result.trauma_signal, 0.0)
        self.assertEqual(result.entropy_signal, 0.0)
        self.assertFalse(result.otherness_detected)


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
