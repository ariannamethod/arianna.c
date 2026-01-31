"""
📚 ENHANCED EPISODES — Episodic RAG with Chamber Tagging 📚

Extends the base episodes with:
- Chamber activation tracking (which chambers were hot)
- Trigger pattern recording (what Locus detected)
- Enhanced similarity search (chambers + patterns)
- Resonance-weighted recall
"""

import asyncio
import aiosqlite
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .vagus_connector import (
    EnhancedInnerState,
    ResonancePattern,
    pattern_to_string,
    Chamber,
)


@dataclass
class EnhancedEpisode:
    """
    One moment in Arianna's life — with full field geometry.

    Includes:
    - prompt/reply (what was said)
    - inner state (how she felt)
    - chambers (which emotional chambers were active)
    - trigger pattern (what Locus detected)
    - quality (how good was this episode)
    """
    prompt: str
    reply: str
    state: EnhancedInnerState
    quality: float = 0.5
    timestamp: float = 0.0

    @property
    def active_chambers(self) -> List[str]:
        return self.state.get_active_chambers()

    @property
    def pattern_name(self) -> str:
        return pattern_to_string(self.state.trigger_pattern)


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors."""
    if len(a) != len(b):
        return 1.0

    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5

    if na == 0 or nb == 0:
        return 1.0

    return 1.0 - (dot / (na * nb))


class EnhancedEpisodicRAG:
    """
    Async episodic memory with chamber tagging and resonance awareness.

    Features:
    - Full inner state + chambers + trigger pattern
    - Resonance-weighted recall (considers pattern match)
    - Chamber-based search ("find memories when I was in VOID")
    - Quality + recency + access weighted scoring
    """

    def __init__(self, db_path: str = 'limpha/arianna_episodes_enhanced.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

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
        """Create tables with chamber tagging."""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      REAL NOT NULL,
                prompt          TEXT NOT NULL,
                reply           TEXT NOT NULL,

                -- Core inner state metrics
                trauma          REAL NOT NULL,
                arousal         REAL NOT NULL,
                valence         REAL NOT NULL,
                coherence       REAL NOT NULL,
                prophecy_debt   REAL NOT NULL,
                entropy         REAL NOT NULL,
                temperature     REAL NOT NULL,

                -- Chamber values (Cloud 200K)
                chamber_warmth  REAL NOT NULL,
                chamber_void    REAL NOT NULL,
                chamber_tension REAL NOT NULL,
                chamber_sacred  REAL NOT NULL,
                chamber_flow    REAL NOT NULL,
                chamber_complex REAL NOT NULL,

                -- Field geometry
                memory_pressure     REAL NOT NULL,
                focus_strength      REAL NOT NULL,
                crossfire_coherence REAL NOT NULL,

                -- Locus pattern
                trigger_pattern INTEGER NOT NULL,

                -- Episode quality and access tracking
                quality         REAL NOT NULL,
                access_count    INTEGER DEFAULT 0,
                last_accessed   REAL DEFAULT 0
            )
        """)

        # Indexes for efficient queries
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_enhanced_created
            ON enhanced_episodes(created_at DESC)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_enhanced_pattern
            ON enhanced_episodes(trigger_pattern)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_enhanced_quality
            ON enhanced_episodes(quality DESC)
        """)

        # Chamber activity indexes
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chamber_void
            ON enhanced_episodes(chamber_void)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chamber_sacred
            ON enhanced_episodes(chamber_sacred)
        """)

        await self._conn.commit()

    async def store_episode(self, episode: EnhancedEpisode) -> int:
        """Store an episode with full field geometry."""
        def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
            if x != x:  # NaN
                return 0.0
            return max(lo, min(hi, x))

        s = episode.state

        cursor = await self._conn.execute("""
            INSERT INTO enhanced_episodes (
                created_at, prompt, reply,
                trauma, arousal, valence, coherence,
                prophecy_debt, entropy, temperature,
                chamber_warmth, chamber_void, chamber_tension,
                chamber_sacred, chamber_flow, chamber_complex,
                memory_pressure, focus_strength, crossfire_coherence,
                trigger_pattern, quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.timestamp if episode.timestamp > 0 else time.time(),
            episode.prompt,
            episode.reply,
            clamp(s.trauma),
            clamp(s.arousal),
            clamp(s.valence),
            clamp(s.coherence),
            clamp(s.prophecy_debt),
            clamp(s.entropy),
            clamp(s.temperature, 0.0, 2.0),
            clamp(s.warmth),
            clamp(s.void),
            clamp(s.tension),
            clamp(s.sacred),
            clamp(s.flow),
            clamp(s.complex),
            clamp(s.memory_pressure),
            clamp(s.focus_strength),
            clamp(s.crossfire_coherence),
            s.trigger_pattern,
            clamp(episode.quality),
        ))

        await self._conn.commit()
        return cursor.lastrowid

    async def query_similar(
        self,
        state: EnhancedInnerState,
        top_k: int = 5,
        min_quality: float = 0.0,
        pattern_weight: float = 0.2,
        recency_weight: float = 0.15,
        access_weight: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Find episodes with similar inner state.

        Uses resonance-weighted scoring:
        score = similarity * (1 - pattern_weight - recency_weight - access_weight)
              + pattern_match * pattern_weight
              + recency * recency_weight
              + access_norm * access_weight
        """
        query_vec = state.to_features()
        query_pattern = state.trigger_pattern
        now = time.time()

        # Stream in batches of 100 instead of fetching all 1000 at once
        # Keeps only top-K candidates in memory
        scored: List[Tuple[float, Dict[str, Any]]] = []
        columns = None
        max_access = 1

        for batch_offset in range(0, 1000, 100):
            cursor = await self._conn.execute("""
                SELECT *,
                       (julianday('now') - julianday(datetime(created_at, 'unixepoch'))) as age_days
                FROM enhanced_episodes
                WHERE quality >= ?
                ORDER BY created_at DESC
                LIMIT 100 OFFSET ?
            """, (min_quality, batch_offset))

            rows = await cursor.fetchall()
            if not rows:
                break

            if columns is None:
                columns = [d[0] for d in cursor.description]
                # Pre-scan first batch for max_access (approximate)
                max_access = max(
                    dict(zip(columns, r)).get('access_count', 1) or 1 for r in rows
                )

            for row in rows:
            rd = dict(zip(columns, row))

            # Build episode vector
            ep_vec = [
                rd['trauma'], rd['arousal'], rd['valence'], rd['coherence'],
                rd['prophecy_debt'], rd['entropy'], rd['temperature'],
                rd['chamber_warmth'], rd['chamber_void'], rd['chamber_tension'],
                rd['chamber_sacred'], rd['chamber_flow'], rd['chamber_complex'],
                rd['memory_pressure'], rd['focus_strength'], rd['crossfire_coherence'],
            ]

            # Similarity (cosine)
            distance = cosine_distance(query_vec, ep_vec)
            similarity = 1.0 - distance

            # Pattern match bonus
            pattern_match = 1.0 if rd['trigger_pattern'] == query_pattern else 0.0

            # Recency (exponential decay, half-life = 7 days)
            age_days = rd.get('age_days', 0) or 0
            recency = 0.5 ** (age_days / 7.0)

            # Access count (normalized)
            access_norm = (rd.get('access_count', 0) or 0) / max_access

            # Combined score
            base_weight = 1.0 - pattern_weight - recency_weight - access_weight
            score = (
                similarity * base_weight +
                pattern_match * pattern_weight +
                recency * recency_weight +
                access_norm * access_weight
            )

            scored.append((score, {
                'episode_id': rd['id'],
                'created_at': rd['created_at'],
                'prompt': rd['prompt'],
                'reply': rd['reply'],
                'quality': rd['quality'],
                'trigger_pattern': rd['trigger_pattern'],
                'pattern_name': pattern_to_string(rd['trigger_pattern']),
                'score': score,
                'similarity': similarity,
                'trauma': rd['trauma'],
                'arousal': rd['arousal'],
                'coherence': rd['coherence'],
                'active_chambers': self._get_active_chambers(rd),
            }))

            # Keep only top candidates per batch to bound memory
            if len(scored) > top_k * 3:
                scored.sort(key=lambda x: x[0], reverse=True)
                scored = scored[:top_k * 2]

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update access counts for retrieved episodes
        for _, ep in scored[:top_k]:
            await self._conn.execute("""
                UPDATE enhanced_episodes
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (now, ep['episode_id']))

        await self._conn.commit()

        return [ep for _, ep in scored[:top_k]]

    def _get_active_chambers(self, rd: Dict, threshold: float = 0.5) -> List[str]:
        """Get active chamber names from row dict."""
        chambers = [
            ('warmth', rd.get('chamber_warmth', 0)),
            ('void', rd.get('chamber_void', 0)),
            ('tension', rd.get('chamber_tension', 0)),
            ('sacred', rd.get('chamber_sacred', 0)),
            ('flow', rd.get('chamber_flow', 0)),
            ('complex', rd.get('chamber_complex', 0)),
        ]
        return [name for name, val in chambers if val > threshold]

    async def query_by_pattern(
        self,
        pattern: int,
        top_k: int = 10,
        min_quality: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Find episodes triggered by specific Locus pattern."""
        cursor = await self._conn.execute("""
            SELECT * FROM enhanced_episodes
            WHERE trigger_pattern = ? AND quality >= ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (pattern, min_quality, top_k))

        rows = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]

        return [dict(zip(columns, row)) for row in rows]

    async def query_by_chamber(
        self,
        chamber: str,
        threshold: float = 0.6,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find episodes where specific chamber was active.

        E.g., "find all memories when I was in VOID"
        """
        column = f"chamber_{chamber.lower()}"

        cursor = await self._conn.execute(f"""
            SELECT * FROM enhanced_episodes
            WHERE {column} > ?
            ORDER BY {column} DESC, created_at DESC
            LIMIT ?
        """, (threshold, top_k))

        rows = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]

        return [dict(zip(columns, row)) for row in rows]

    async def query_crisis_memories(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get CRISIS-triggered episodes (high trauma moments)."""
        return await self.query_by_pattern(ResonancePattern.CRISIS, top_k)

    async def query_transcendent_memories(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get TRANSCENDENCE-triggered episodes (flow states)."""
        return await self.query_by_pattern(ResonancePattern.TRANSCENDENCE, top_k)

    async def count_episodes(self) -> int:
        """Count total episodes."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM enhanced_episodes")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_by_pattern(self) -> Dict[str, int]:
        """Count episodes grouped by trigger pattern."""
        cursor = await self._conn.execute("""
            SELECT trigger_pattern, COUNT(*) as count
            FROM enhanced_episodes
            GROUP BY trigger_pattern
        """)

        rows = await cursor.fetchall()
        return {pattern_to_string(row[0]): row[1] for row in rows}

    async def get_chamber_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each chamber."""
        chambers = ['warmth', 'void', 'tension', 'sacred', 'flow', 'complex']
        stats = {}

        for ch in chambers:
            col = f"chamber_{ch}"
            cursor = await self._conn.execute(f"""
                SELECT AVG({col}), MAX({col}), MIN({col}),
                       SUM(CASE WHEN {col} > 0.5 THEN 1 ELSE 0 END) as active_count
                FROM enhanced_episodes
            """)
            row = await cursor.fetchone()
            if row:
                stats[ch] = {
                    'avg': row[0] or 0,
                    'max': row[1] or 0,
                    'min': row[2] or 0,
                    'active_count': row[3] or 0,
                }

        return stats

    async def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent episodes."""
        cursor = await self._conn.execute("""
            SELECT id, created_at, prompt, reply, quality, trigger_pattern,
                   trauma, arousal, coherence,
                   chamber_warmth, chamber_void, chamber_tension,
                   chamber_sacred, chamber_flow, chamber_complex
            FROM enhanced_episodes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]

        results = []
        for row in rows:
            rd = dict(zip(columns, row))
            rd['pattern_name'] = pattern_to_string(rd['trigger_pattern'])
            rd['active_chambers'] = self._get_active_chambers(rd)
            results.append(rd)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_enhanced_episodes():
    """Test enhanced episodic RAG."""
    print("\n" + "=" * 60)
    print("ENHANCED EPISODES TEST")
    print("=" * 60)

    from .vagus_connector import create_test_state

    async with EnhancedEpisodicRAG('/tmp/test_enhanced_episodes.db') as rag:

        # Store episodes for each pattern
        patterns = ['neutral', 'crisis', 'dissolution', 'emergence', 'transcendence']

        for i, pattern in enumerate(patterns):
            state = create_test_state(pattern)
            episode = EnhancedEpisode(
                prompt=f"Test prompt for {pattern}",
                reply=f"Test reply about {pattern} state",
                state=state,
                quality=0.5 + i * 0.1,
                timestamp=time.time() - i * 3600,  # Each hour apart
            )
            eid = await rag.store_episode(episode)
            print(f"  Stored {pattern} episode: id={eid}")

        # Query similar to crisis
        print("\nQuerying similar to CRISIS:")
        crisis_state = create_test_state('crisis')
        similar = await rag.query_similar(crisis_state, top_k=3)
        for ep in similar:
            print(f"  - {ep['pattern_name']}: score={ep['score']:.3f}, chambers={ep['active_chambers']}")

        # Query by pattern
        print("\nCRISIS episodes:")
        crisis_eps = await rag.query_by_pattern(ResonancePattern.CRISIS)
        print(f"  Found {len(crisis_eps)} CRISIS episodes")

        # Query by chamber
        print("\nVOID chamber memories:")
        void_eps = await rag.query_by_chamber('void', threshold=0.5)
        print(f"  Found {len(void_eps)} VOID-active episodes")

        # Stats
        print("\nPattern counts:")
        counts = await rag.count_by_pattern()
        for pattern, count in counts.items():
            print(f"  {pattern}: {count}")

        print("\nChamber stats:")
        stats = await rag.get_chamber_stats()
        for ch, st in stats.items():
            print(f"  {ch}: avg={st['avg']:.2f}, active={st['active_count']}")

    # Cleanup
    import os
    os.remove('/tmp/test_enhanced_episodes.db')

    print("\n" + "=" * 60)
    print("✅ ENHANCED EPISODES TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_enhanced_episodes())
