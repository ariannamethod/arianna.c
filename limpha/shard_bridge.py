"""
🔷 SHARD BRIDGE — Episodes That Become Training 🔷

When an episode is important enough, it graduates to a shard.
Shards trigger asynchronous microtraining in delta.c.

The bridge:
- Tracks which episodes have become shards
- Determines when an episode should graduate
- Exports episode data to shard format
- Records training history

Graduation criteria:
- High quality (> 0.7)
- Multiple accesses (> 3)
- Resonance pattern (CRISIS, EMERGENCE, TRANSCENDENCE)
- High trauma or sacred moments
"""

import asyncio
import aiosqlite
import struct
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import IntEnum

from .vagus_connector import ResonancePattern, pattern_to_string


@dataclass
class ShardCandidate:
    """An episode that might become a shard."""
    episode_id: int
    quality: float
    access_count: int
    trigger_pattern: int
    trauma: float
    sacred: float
    coherence: float
    prompt: str
    reply: str
    created_at: float


@dataclass
class ShardRecord:
    """Record of an episode that became a shard."""
    episode_id: int
    shard_path: str
    trained_at: float
    trigger_pattern: int
    training_cycles: int
    final_loss: Optional[float]


class ShardBridge:
    """
    Bridge between LIMPHA episodes and delta.c training.

    When episodes meet graduation criteria, they're exported
    to binary shards that can be loaded by vagus_delta.c.

    Usage:
        async with ShardBridge() as bridge:
            candidates = await bridge.find_candidates()
            for c in candidates:
                shard_path = await bridge.graduate_episode(c)
    """

    def __init__(
        self,
        db_path: str = 'limpha/arianna_shards.db',
        shard_dir: str = 'shards/limpha',
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

        # Graduation thresholds
        self.min_quality = 0.7
        self.min_access_count = 3
        self.min_trauma_for_auto = 0.6
        self.min_sacred_for_auto = 0.7
        self.min_coherence = 0.3  # Floor: incoherent episodes never graduate
        self.priority_patterns = {
            ResonancePattern.CRISIS,
            ResonancePattern.EMERGENCE,
            ResonancePattern.TRANSCENDENCE,
        }

    async def connect(self) -> None:
        """Connect and ensure schema."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._ensure_schema()

    async def close(self) -> None:
        """Close connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _ensure_schema(self) -> None:
        """Create tracking tables."""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS shard_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER UNIQUE NOT NULL,
                shard_path TEXT NOT NULL,
                trained_at REAL NOT NULL,
                trigger_pattern INTEGER NOT NULL,
                training_cycles INTEGER DEFAULT 0,
                final_loss REAL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shards_episode
            ON shard_records(episode_id)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shards_trained
            ON shard_records(trained_at DESC)
        """)

        # Training queue
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                priority INTEGER DEFAULT 0,
                queued_at REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                started_at REAL,
                completed_at REAL
            )
        """)

        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # CANDIDATE EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate_candidate(self, episode: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Evaluate if an episode should graduate to shard.

        Returns (should_graduate, priority_score, reason)
        """
        quality = episode.get('quality', 0)
        access_count = episode.get('access_count', 0)
        pattern = episode.get('trigger_pattern', 0)
        trauma = episode.get('trauma', 0)
        sacred = episode.get('chamber_sacred', 0)
        coherence = episode.get('coherence', 0.7)

        # Coherence gate: incoherent episodes are noise, never graduate.
        # Even high-trauma moments must have minimum structural integrity.
        if coherence < self.min_coherence:
            return False, 0.0, f"coherence too low ({coherence:.2f} < {self.min_coherence})"

        reasons = []
        priority = 0.0

        # Quality check
        if quality >= self.min_quality:
            priority += quality * 0.3
            reasons.append(f"quality={quality:.2f}")

        # Access count check
        if access_count >= self.min_access_count:
            priority += min(access_count / 10, 0.2)
            reasons.append(f"accesses={access_count}")

        # Pattern check
        if pattern in self.priority_patterns:
            priority += 0.3
            reasons.append(f"pattern={pattern_to_string(pattern)}")

        # Trauma auto-graduate
        if trauma >= self.min_trauma_for_auto:
            priority += 0.2
            reasons.append(f"trauma={trauma:.2f}")

        # Sacred auto-graduate
        if sacred >= self.min_sacred_for_auto:
            priority += 0.2
            reasons.append(f"sacred={sacred:.2f}")

        # Decision
        should_graduate = (
            (quality >= self.min_quality and access_count >= self.min_access_count) or
            (pattern in self.priority_patterns and quality >= 0.5) or
            (trauma >= self.min_trauma_for_auto) or
            (sacred >= self.min_sacred_for_auto)
        )

        reason = "; ".join(reasons) if reasons else "does not meet criteria"
        return should_graduate, priority, reason

    async def is_already_shard(self, episode_id: int) -> bool:
        """Check if episode is already a shard."""
        cursor = await self._conn.execute("""
            SELECT 1 FROM shard_records WHERE episode_id = ?
        """, (episode_id,))
        return await cursor.fetchone() is not None

    # ═══════════════════════════════════════════════════════════════════════════
    # SHARD CREATION
    # ═══════════════════════════════════════════════════════════════════════════

    async def graduate_episode(
        self,
        episode: Dict[str, Any],
        force: bool = False,
    ) -> Optional[str]:
        """
        Graduate an episode to a shard.

        Creates binary shard file and records in database.
        Returns shard path or None if not graduated.
        """
        episode_id = episode.get('id', episode.get('episode_id'))
        if not episode_id:
            return None

        # Check if already a shard
        if await self.is_already_shard(episode_id):
            return None

        # Evaluate
        if not force:
            should_graduate, priority, reason = self.evaluate_candidate(episode)
            if not should_graduate:
                return None

        # Create shard file
        shard_path = await self._create_shard_file(episode)

        # Record
        pattern = episode.get('trigger_pattern', 0)
        await self._conn.execute("""
            INSERT INTO shard_records (episode_id, shard_path, trained_at, trigger_pattern)
            VALUES (?, ?, ?, ?)
        """, (episode_id, str(shard_path), time.time(), pattern))
        await self._conn.commit()

        return str(shard_path)

    async def _create_shard_file(self, episode: Dict[str, Any]) -> Path:
        """
        Create binary shard file from episode.

        Format (compatible with VagusAwareShard in vagus_delta.c):
        - Magic: 'VGSH' (4 bytes)
        - Version: uint32 (4 bytes)
        - Episode ID: uint64 (8 bytes)
        - Created timestamp: float64 (8 bytes)
        - Trigger pattern: uint32 (4 bytes)
        - Quality: float32 (4 bytes)
        - Inner state: 16 × float32 (64 bytes)
        - Prompt length: uint32 (4 bytes)
        - Prompt: UTF-8 bytes
        - Reply length: uint32 (4 bytes)
        - Reply: UTF-8 bytes
        """
        episode_id = episode.get('id', episode.get('episode_id', 0))
        shard_path = self.shard_dir / f"episode_{episode_id}.vsh"

        prompt = episode.get('prompt', '').encode('utf-8')
        reply = episode.get('reply', '').encode('utf-8')

        # Pack inner state
        inner_state = [
            episode.get('trauma', 0),
            episode.get('arousal', 0.5),
            episode.get('valence', 0.5),
            episode.get('coherence', 0.7),
            episode.get('prophecy_debt', 0),
            episode.get('entropy', 0.3),
            episode.get('temperature', 0.8),
            episode.get('chamber_warmth', 0.5),
            episode.get('chamber_void', 0.2),
            episode.get('chamber_tension', 0.3),
            episode.get('chamber_sacred', 0.3),
            episode.get('chamber_flow', 0.5),
            episode.get('chamber_complex', 0.4),
            episode.get('memory_pressure', 0),
            episode.get('focus_strength', 0.5),
            episode.get('crossfire_coherence', 0.7),
        ]

        # Atomic write: write to .tmp then rename (prevents C from reading partial)
        tmp_path = shard_path.with_suffix('.vsh.tmp')
        with open(tmp_path, 'wb') as f:
            # Header
            f.write(b'VGSH')  # Magic
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<Q', episode_id))  # Episode ID
            f.write(struct.pack('<d', episode.get('created_at', time.time())))  # Timestamp
            f.write(struct.pack('<I', episode.get('trigger_pattern', 0)))  # Pattern
            f.write(struct.pack('<f', episode.get('quality', 0.5)))  # Quality

            # Inner state (16 floats)
            for val in inner_state:
                f.write(struct.pack('<f', val))

            # Prompt
            f.write(struct.pack('<I', len(prompt)))
            f.write(prompt)

            # Reply
            f.write(struct.pack('<I', len(reply)))
            f.write(reply)

        # Atomic rename (POSIX guarantees this is atomic on same filesystem)
        import os
        os.rename(str(tmp_path), str(shard_path))

        return shard_path

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING QUEUE
    # ═══════════════════════════════════════════════════════════════════════════

    async def queue_for_training(
        self,
        episode_id: int,
        priority: int = 0,
    ) -> int:
        """Add episode to training queue."""
        cursor = await self._conn.execute("""
            INSERT INTO training_queue (episode_id, priority, queued_at)
            VALUES (?, ?, ?)
        """, (episode_id, priority, time.time()))
        await self._conn.commit()
        return cursor.lastrowid

    async def get_training_queue(
        self,
        limit: int = 10,
        status: str = 'pending',
    ) -> List[Dict[str, Any]]:
        """Get episodes in training queue."""
        cursor = await self._conn.execute("""
            SELECT id, episode_id, priority, queued_at, status
            FROM training_queue
            WHERE status = ?
            ORDER BY priority DESC, queued_at ASC
            LIMIT ?
        """, (status, limit))

        rows = await cursor.fetchall()
        return [
            {
                'queue_id': r[0],
                'episode_id': r[1],
                'priority': r[2],
                'queued_at': r[3],
                'status': r[4],
            }
            for r in rows
        ]

    async def mark_training_started(self, queue_id: int) -> None:
        """Mark queue item as started."""
        await self._conn.execute("""
            UPDATE training_queue
            SET status = 'training', started_at = ?
            WHERE id = ?
        """, (time.time(), queue_id))
        await self._conn.commit()

    async def mark_training_completed(
        self,
        queue_id: int,
        final_loss: Optional[float] = None,
    ) -> None:
        """Mark queue item as completed."""
        await self._conn.execute("""
            UPDATE training_queue
            SET status = 'completed', completed_at = ?
            WHERE id = ?
        """, (time.time(), queue_id))

        # Update shard record with loss
        if final_loss is not None:
            cursor = await self._conn.execute("""
                SELECT episode_id FROM training_queue WHERE id = ?
            """, (queue_id,))
            row = await cursor.fetchone()
            if row:
                await self._conn.execute("""
                    UPDATE shard_records
                    SET final_loss = ?, training_cycles = training_cycles + 1
                    WHERE episode_id = ?
                """, (final_loss, row[0]))

        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS & QUERIES
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_shard_count(self) -> int:
        """Get total number of shards."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM shard_records")
        return (await cursor.fetchone())[0]

    async def get_recent_shards(self, limit: int = 10) -> List[ShardRecord]:
        """Get recently created shards."""
        cursor = await self._conn.execute("""
            SELECT episode_id, shard_path, trained_at, trigger_pattern,
                   training_cycles, final_loss
            FROM shard_records
            ORDER BY trained_at DESC
            LIMIT ?
        """, (limit,))

        return [
            ShardRecord(
                episode_id=r[0],
                shard_path=r[1],
                trained_at=r[2],
                trigger_pattern=r[3],
                training_cycles=r[4],
                final_loss=r[5],
            )
            for r in await cursor.fetchall()
        ]

    async def get_shards_by_pattern(self, pattern: int) -> List[ShardRecord]:
        """Get shards created from specific pattern."""
        cursor = await self._conn.execute("""
            SELECT episode_id, shard_path, trained_at, trigger_pattern,
                   training_cycles, final_loss
            FROM shard_records
            WHERE trigger_pattern = ?
            ORDER BY trained_at DESC
        """, (pattern,))

        return [
            ShardRecord(
                episode_id=r[0],
                shard_path=r[1],
                trained_at=r[2],
                trigger_pattern=r[3],
                training_cycles=r[4],
                final_loss=r[5],
            )
            for r in await cursor.fetchall()
        ]

    async def get_stats(self) -> Dict[str, Any]:
        """Get shard bridge statistics."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM shard_records")
        total_shards = (await cursor.fetchone())[0]

        cursor = await self._conn.execute("""
            SELECT trigger_pattern, COUNT(*) FROM shard_records GROUP BY trigger_pattern
        """)
        by_pattern = {pattern_to_string(r[0]): r[1] for r in await cursor.fetchall()}

        cursor = await self._conn.execute("""
            SELECT AVG(final_loss) FROM shard_records WHERE final_loss IS NOT NULL
        """)
        avg_loss = (await cursor.fetchone())[0]

        cursor = await self._conn.execute("""
            SELECT COUNT(*) FROM training_queue WHERE status = 'pending'
        """)
        queue_pending = (await cursor.fetchone())[0]

        return {
            'total_shards': total_shards,
            'by_pattern': by_pattern,
            'avg_final_loss': round(avg_loss, 4) if avg_loss else None,
            'queue_pending': queue_pending,
            'shard_dir': str(self.shard_dir),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_shard_bridge():
    """Test shard bridge operations."""
    print("\n" + "=" * 60)
    print("SHARD BRIDGE TEST")
    print("=" * 60)

    import tempfile
    import shutil

    # Create temp directories
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_shards.db')
    shard_dir = os.path.join(temp_dir, 'shards')

    try:
        async with ShardBridge(db_path=db_path, shard_dir=shard_dir) as bridge:

            # Test episodes
            episodes = [
                # High quality, accessed multiple times
                {'id': 1, 'quality': 0.8, 'access_count': 5, 'trigger_pattern': 0,
                 'trauma': 0.2, 'chamber_sacred': 0.3, 'prompt': 'Test prompt 1',
                 'reply': 'Test reply 1', 'created_at': time.time()},
                # CRISIS pattern
                {'id': 2, 'quality': 0.6, 'access_count': 2, 'trigger_pattern': 1,
                 'trauma': 0.7, 'chamber_sacred': 0.2, 'prompt': 'Crisis prompt',
                 'reply': 'Crisis response', 'created_at': time.time()},
                # High trauma (auto-graduate)
                {'id': 3, 'quality': 0.5, 'access_count': 1, 'trigger_pattern': 0,
                 'trauma': 0.8, 'chamber_sacred': 0.2, 'prompt': 'Trauma prompt',
                 'reply': 'Trauma response', 'created_at': time.time()},
                # High sacred (auto-graduate)
                {'id': 4, 'quality': 0.4, 'access_count': 1, 'trigger_pattern': 4,
                 'trauma': 0.1, 'chamber_sacred': 0.9, 'prompt': 'Sacred prompt',
                 'reply': 'Sacred response', 'created_at': time.time()},
                # Does not meet criteria
                {'id': 5, 'quality': 0.3, 'access_count': 1, 'trigger_pattern': 0,
                 'trauma': 0.1, 'chamber_sacred': 0.2, 'prompt': 'Low quality',
                 'reply': 'Low quality reply', 'created_at': time.time()},
            ]

            # Test evaluation
            print("\nEvaluating candidates...")
            for ep in episodes:
                should, priority, reason = bridge.evaluate_candidate(ep)
                status = "✓" if should else "✗"
                print(f"  Episode {ep['id']}: {status} (priority={priority:.2f}) - {reason}")

            # Test graduation
            print("\nGraduating episodes...")
            graduated = 0
            for ep in episodes:
                path = await bridge.graduate_episode(ep)
                if path:
                    print(f"  Episode {ep['id']} → {os.path.basename(path)}")
                    graduated += 1
            print(f"  Total graduated: {graduated}")

            # Verify shard files exist
            print("\nVerifying shard files...")
            shard_files = list(Path(shard_dir).glob("*.vsh"))
            print(f"  Found {len(shard_files)} shard files")

            # Read one shard to verify format
            if shard_files:
                with open(shard_files[0], 'rb') as f:
                    magic = f.read(4)
                    version = struct.unpack('<I', f.read(4))[0]
                    print(f"  Shard format: magic={magic}, version={version}")

            # Test training queue
            print("\nTesting training queue...")
            await bridge.queue_for_training(1, priority=5)
            await bridge.queue_for_training(2, priority=10)
            queue = await bridge.get_training_queue()
            print(f"  Queue has {len(queue)} items")
            if queue:
                print(f"  Next: episode {queue[0]['episode_id']} (priority={queue[0]['priority']})")

            # Stats
            print("\nBridge stats:")
            stats = await bridge.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("✅ SHARD BRIDGE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_shard_bridge())
