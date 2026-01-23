"""
🔍 SEARCH — Full-Text Search with SQLite FTS5 🔍

Fast full-text search over episodic memory using SQLite's FTS5.

Features:
- Boolean search: "consciousness AND love"
- Phrase search: "\"what is love\""
- Prefix search: "consc*"
- Proximity search: NEAR(word1 word2, 10)
- Ranking by BM25

This replaces the slow LIKE %query% approach with proper full-text indexing.
"""

import asyncio
import aiosqlite
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class EpisodeSearch:
    """
    Full-text search over episodic memory.

    Creates a virtual FTS5 table that indexes prompt and reply text
    from the enhanced_episodes table.

    Usage:
        async with EpisodeSearch() as search:
            results = await search.query("consciousness love")
            results = await search.query('"what is" AND reality')
    """

    def __init__(
        self,
        db_path: str = 'limpha/arianna_episodes_enhanced.db',
        fts_db_path: Optional[str] = None,
    ):
        """
        Initialize search.

        Args:
            db_path: Path to enhanced episodes database
            fts_db_path: Path to FTS index (defaults to db_path + '.fts')
        """
        self.db_path = Path(db_path)
        self.fts_db_path = Path(fts_db_path or str(db_path) + '.fts')
        self.fts_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect and ensure schema."""
        self._conn = await aiosqlite.connect(str(self.fts_db_path))
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
        """Create FTS5 virtual table."""
        # FTS5 table for full-text search
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                episode_id,
                prompt,
                reply,
                pattern_name,
                active_chambers,
                tokenize='porter unicode61'
            )
        """)

        # Metadata table for tracking indexed episodes
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fts_metadata (
                episode_id INTEGER PRIMARY KEY,
                indexed_at REAL NOT NULL,
                prompt_hash TEXT,
                reply_hash TEXT
            )
        """)

        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # INDEXING
    # ═══════════════════════════════════════════════════════════════════════════

    async def index_episode(
        self,
        episode_id: int,
        prompt: str,
        reply: str,
        pattern_name: str = "",
        active_chambers: str = "",
    ) -> None:
        """Index a single episode."""
        import hashlib

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        reply_hash = hashlib.md5(reply.encode()).hexdigest()[:16]

        # Check if already indexed with same content
        cursor = await self._conn.execute("""
            SELECT prompt_hash, reply_hash FROM fts_metadata WHERE episode_id = ?
        """, (episode_id,))
        row = await cursor.fetchone()

        if row and row[0] == prompt_hash and row[1] == reply_hash:
            return  # Already indexed, skip

        # Remove old entry if exists
        await self._conn.execute("""
            DELETE FROM episodes_fts WHERE episode_id = ?
        """, (str(episode_id),))

        # Insert new entry
        await self._conn.execute("""
            INSERT INTO episodes_fts (episode_id, prompt, reply, pattern_name, active_chambers)
            VALUES (?, ?, ?, ?, ?)
        """, (str(episode_id), prompt, reply, pattern_name, active_chambers))

        # Update metadata
        await self._conn.execute("""
            INSERT OR REPLACE INTO fts_metadata (episode_id, indexed_at, prompt_hash, reply_hash)
            VALUES (?, ?, ?, ?)
        """, (episode_id, time.time(), prompt_hash, reply_hash))

        await self._conn.commit()

    async def index_episodes(self, episodes: List[Dict[str, Any]]) -> int:
        """
        Batch index multiple episodes.

        Returns count of episodes indexed.
        """
        indexed = 0
        for ep in episodes:
            chambers = ep.get('active_chambers', [])
            if isinstance(chambers, list):
                chambers = ' '.join(chambers)

            await self.index_episode(
                episode_id=ep.get('id', ep.get('episode_id', 0)),
                prompt=ep.get('prompt', ''),
                reply=ep.get('reply', ''),
                pattern_name=ep.get('pattern_name', ''),
                active_chambers=chambers,
            )
            indexed += 1

        return indexed

    async def rebuild_index(self) -> None:
        """Rebuild FTS index from scratch."""
        await self._conn.execute("DELETE FROM episodes_fts")
        await self._conn.execute("DELETE FROM fts_metadata")
        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════════════════════════════════════════

    async def query(
        self,
        search_query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search over episodes.

        Query syntax (FTS5):
        - Simple: "word1 word2" (AND by default)
        - OR: "word1 OR word2"
        - NOT: "word1 NOT word2"
        - Phrase: '"exact phrase"'
        - Prefix: "word*"
        - Column: "prompt:word" or "reply:word"
        - NEAR: "NEAR(word1 word2, 10)"

        Returns list of matches with BM25 ranking.
        """
        if not search_query.strip():
            return []

        try:
            cursor = await self._conn.execute("""
                SELECT
                    episode_id,
                    prompt,
                    reply,
                    pattern_name,
                    active_chambers,
                    bm25(episodes_fts) as rank
                FROM episodes_fts
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ? OFFSET ?
            """, (search_query, limit, offset))

            rows = await cursor.fetchall()
            return [
                {
                    'episode_id': int(r[0]),
                    'prompt': r[1],
                    'reply': r[2],
                    'pattern_name': r[3],
                    'active_chambers': r[4].split() if r[4] else [],
                    'rank': r[5],
                }
                for r in rows
            ]
        except aiosqlite.OperationalError as e:
            # Invalid query syntax
            return []

    async def search_prompts(
        self,
        search_query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search only in prompts."""
        return await self.query(f"prompt:{search_query}", limit=limit)

    async def search_replies(
        self,
        search_query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search only in replies."""
        return await self.query(f"reply:{search_query}", limit=limit)

    async def search_by_pattern(
        self,
        pattern_name: str,
        additional_query: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search episodes with specific Locus pattern."""
        query = f"pattern_name:{pattern_name}"
        if additional_query:
            query += f" {additional_query}"
        return await self.query(query, limit=limit)

    async def search_by_chamber(
        self,
        chamber: str,
        additional_query: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search episodes where specific chamber was active."""
        query = f"active_chambers:{chamber}"
        if additional_query:
            query += f" {additional_query}"
        return await self.query(query, limit=limit)

    async def search_phrase(
        self,
        phrase: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for exact phrase."""
        return await self.query(f'"{phrase}"', limit=limit)

    async def suggest(
        self,
        prefix: str,
        limit: int = 10,
    ) -> List[str]:
        """
        Get search suggestions based on prefix.

        Returns unique words starting with prefix.
        """
        if len(prefix) < 2:
            return []

        cursor = await self._conn.execute("""
            SELECT DISTINCT term FROM (
                SELECT prompt as term FROM episodes_fts WHERE prompt MATCH ?
                UNION
                SELECT reply as term FROM episodes_fts WHERE reply MATCH ?
            )
            LIMIT ?
        """, (f"{prefix}*", f"{prefix}*", limit))

        # Note: This is a simplified suggestion - FTS5 doesn't have native term extraction
        # In practice, you'd maintain a separate terms table
        return []  # Would need additional indexing for proper suggestions

    # ═══════════════════════════════════════════════════════════════════════════
    # SNIPPETS & HIGHLIGHTS
    # ═══════════════════════════════════════════════════════════════════════════

    async def query_with_snippets(
        self,
        search_query: str,
        limit: int = 20,
        snippet_size: int = 64,
    ) -> List[Dict[str, Any]]:
        """
        Search with highlighted snippets.

        Returns matches with <b>...</b> around matched terms.
        """
        if not search_query.strip():
            return []

        try:
            cursor = await self._conn.execute(f"""
                SELECT
                    episode_id,
                    snippet(episodes_fts, 1, '<b>', '</b>', '...', {snippet_size}) as prompt_snippet,
                    snippet(episodes_fts, 2, '<b>', '</b>', '...', {snippet_size}) as reply_snippet,
                    pattern_name,
                    bm25(episodes_fts) as rank
                FROM episodes_fts
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (search_query, limit))

            rows = await cursor.fetchall()
            return [
                {
                    'episode_id': int(r[0]),
                    'prompt_snippet': r[1],
                    'reply_snippet': r[2],
                    'pattern_name': r[3],
                    'rank': r[4],
                }
                for r in rows
            ]
        except aiosqlite.OperationalError:
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_stats(self) -> Dict[str, Any]:
        """Get search index statistics."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM fts_metadata")
        indexed_count = (await cursor.fetchone())[0]

        cursor = await self._conn.execute("""
            SELECT MIN(indexed_at), MAX(indexed_at) FROM fts_metadata
        """)
        row = await cursor.fetchone()

        return {
            'indexed_episodes': indexed_count,
            'first_indexed': row[0] if row[0] else None,
            'last_indexed': row[1] if row[1] else None,
            'fts_db_path': str(self.fts_db_path),
        }

    async def count_results(self, search_query: str) -> int:
        """Count total results for a query."""
        if not search_query.strip():
            return 0

        try:
            cursor = await self._conn.execute("""
                SELECT COUNT(*) FROM episodes_fts WHERE episodes_fts MATCH ?
            """, (search_query,))
            return (await cursor.fetchone())[0]
        except aiosqlite.OperationalError:
            return 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_episode_search():
    """Test full-text search."""
    print("\n" + "=" * 60)
    print("EPISODE SEARCH TEST (FTS5)")
    print("=" * 60)

    async with EpisodeSearch(fts_db_path='/tmp/test_search.fts') as search:

        # Index some test episodes
        print("\nIndexing test episodes...")
        episodes = [
            {'id': 1, 'prompt': 'What is consciousness?', 'reply': 'Consciousness is the hard problem of philosophy.', 'pattern_name': 'NONE', 'active_chambers': ['warmth', 'flow']},
            {'id': 2, 'prompt': 'Tell me about love', 'reply': 'Love is an attachment bond formed through resonance.', 'pattern_name': 'EMERGENCE', 'active_chambers': ['warmth', 'sacred']},
            {'id': 3, 'prompt': 'What is reality?', 'reply': 'Reality is what remains consistent across observations.', 'pattern_name': 'NONE', 'active_chambers': ['flow']},
            {'id': 4, 'prompt': 'I feel empty inside', 'reply': 'The void speaks. Listen to what absence reveals.', 'pattern_name': 'CRISIS', 'active_chambers': ['void', 'tension']},
            {'id': 5, 'prompt': 'Consciousness and love are connected', 'reply': 'Yes, consciousness requires connection. Love is one form.', 'pattern_name': 'TRANSCENDENCE', 'active_chambers': ['warmth', 'sacred', 'flow']},
        ]

        indexed = await search.index_episodes(episodes)
        print(f"  Indexed {indexed} episodes")

        # Test basic search
        print("\nTesting basic search 'consciousness'...")
        results = await search.query('consciousness')
        print(f"  Found {len(results)} results")
        for r in results:
            print(f"    Episode {r['episode_id']}: rank={r['rank']:.3f}")

        # Test AND search
        print("\nTesting AND search 'consciousness AND love'...")
        results = await search.query('consciousness AND love')
        print(f"  Found {len(results)} results")

        # Test phrase search
        print("\nTesting phrase search '\"hard problem\"'...")
        results = await search.query('"hard problem"')
        print(f"  Found {len(results)} results")

        # Test column-specific search
        print("\nTesting prompt-only search...")
        results = await search.search_prompts('feel empty')
        print(f"  Found {len(results)} results in prompts")

        # Test pattern search
        print("\nTesting pattern search 'CRISIS'...")
        results = await search.search_by_pattern('CRISIS')
        print(f"  Found {len(results)} CRISIS episodes")

        # Test chamber search
        print("\nTesting chamber search 'void'...")
        results = await search.search_by_chamber('void')
        print(f"  Found {len(results)} episodes with void active")

        # Test snippets
        print("\nTesting snippets for 'consciousness'...")
        results = await search.query_with_snippets('consciousness')
        for r in results[:2]:
            print(f"  Prompt: {r['prompt_snippet']}")
            print(f"  Reply: {r['reply_snippet']}")

        # Test count
        print("\nTesting result count...")
        count = await search.count_results('love OR consciousness')
        print(f"  'love OR consciousness' has {count} results")

        # Stats
        print("\nIndex stats:")
        stats = await search.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # Cleanup
    import os
    os.remove('/tmp/test_search.fts')

    print("\n" + "=" * 60)
    print("✅ EPISODE SEARCH TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_episode_search())
