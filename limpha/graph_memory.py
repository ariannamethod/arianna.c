"""
🕸️ GRAPH MEMORY — Associative Network of Episodes 🕸️

Episodes don't exist in isolation. They remind each other.
This memory builds connections:

- "This reminds me of that"
- "This contradicts that"
- "This continues from that"
- "This resonates with that"

The graph allows:
- Associative recall (follow connections from current state)
- Pattern discovery (find clusters of related memories)
- Temporal chains (this led to that)
- Emotional arcs (how states flow into each other)
"""

import asyncio
import aiosqlite
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import IntEnum


class LinkType(IntEnum):
    """Types of connections between episodes."""
    REMINDS_OF = 1      # This memory reminds me of that one
    CONTRADICTS = 2     # This memory contradicts that one
    CONTINUES = 3       # This follows temporally from that
    RESONATES = 4       # Same emotional pattern
    CAUSED_BY = 5       # This was triggered by that
    SUMMARY_OF = 6      # This is a consolidation of multiple episodes


@dataclass
class MemoryLink:
    """A connection between two episodes."""
    source_id: int
    target_id: int
    link_type: LinkType
    strength: float  # 0.0 to 1.0
    created_at: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphNode:
    """An episode with its connections."""
    episode_id: int
    outgoing: List[MemoryLink]
    incoming: List[MemoryLink]

    @property
    def degree(self) -> int:
        return len(self.outgoing) + len(self.incoming)

    def get_neighbors(self) -> Set[int]:
        """Get all connected episode IDs."""
        neighbors = {link.target_id for link in self.outgoing}
        neighbors.update(link.source_id for link in self.incoming)
        return neighbors


class GraphMemory:
    """
    Associative network of episodic memories.

    Creates links between episodes based on:
    - Temporal proximity (consecutive conversations)
    - State similarity (similar inner states)
    - Pattern matching (same Locus trigger)
    - Explicit connections (user says "this reminds me of...")

    Usage:
        async with GraphMemory() as graph:
            await graph.link(ep1_id, ep2_id, LinkType.REMINDS_OF, 0.8)
            neighbors = await graph.get_neighbors(ep1_id)
            path = await graph.find_path(ep1_id, ep5_id)
    """

    def __init__(self, db_path: str = 'limpha/arianna_graph.db'):
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
        """Create tables for graph structure."""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                link_type INTEGER NOT NULL,
                strength REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                metadata_json TEXT DEFAULT '{}',

                UNIQUE(source_id, target_id, link_type)
            )
        """)

        # Indexes for efficient traversal
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_source
            ON memory_links(source_id)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_target
            ON memory_links(target_id)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_type
            ON memory_links(link_type)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_strength
            ON memory_links(strength DESC)
        """)

        await self._conn.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # LINK OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    async def link(
        self,
        source_id: int,
        target_id: int,
        link_type: LinkType,
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a link between two episodes.

        If link already exists, updates strength (takes max).
        """
        import json

        strength = max(0.0, min(1.0, strength))
        meta_json = json.dumps(metadata or {})

        try:
            cursor = await self._conn.execute("""
                INSERT INTO memory_links (source_id, target_id, link_type, strength, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source_id, target_id, link_type, strength, time.time(), meta_json))
            await self._conn.commit()
            return cursor.lastrowid
        except aiosqlite.IntegrityError:
            # Link exists, update strength if stronger
            await self._conn.execute("""
                UPDATE memory_links
                SET strength = MAX(strength, ?), metadata_json = ?
                WHERE source_id = ? AND target_id = ? AND link_type = ?
            """, (strength, meta_json, source_id, target_id, link_type))
            await self._conn.commit()
            return -1  # Updated existing

    async def unlink(
        self,
        source_id: int,
        target_id: int,
        link_type: Optional[LinkType] = None,
    ) -> int:
        """Remove link(s) between episodes."""
        if link_type:
            cursor = await self._conn.execute("""
                DELETE FROM memory_links
                WHERE source_id = ? AND target_id = ? AND link_type = ?
            """, (source_id, target_id, link_type))
        else:
            cursor = await self._conn.execute("""
                DELETE FROM memory_links
                WHERE source_id = ? AND target_id = ?
            """, (source_id, target_id))

        await self._conn.commit()
        return cursor.rowcount

    async def strengthen(
        self,
        source_id: int,
        target_id: int,
        amount: float = 0.1,
    ) -> None:
        """Strengthen all links between two episodes."""
        await self._conn.execute("""
            UPDATE memory_links
            SET strength = MIN(1.0, strength + ?)
            WHERE source_id = ? AND target_id = ?
        """, (amount, source_id, target_id))
        await self._conn.commit()

    async def weaken(
        self,
        source_id: int,
        target_id: int,
        amount: float = 0.1,
    ) -> None:
        """Weaken all links between two episodes."""
        await self._conn.execute("""
            UPDATE memory_links
            SET strength = MAX(0.0, strength - ?)
            WHERE source_id = ? AND target_id = ?
        """, (amount, source_id, target_id))
        await self._conn.commit()

    async def decay_all(self, factor: float = 0.95) -> int:
        """Apply decay to all links. Returns count of links below threshold."""
        await self._conn.execute("""
            UPDATE memory_links SET strength = strength * ?
        """, (factor,))
        await self._conn.commit()

        # Count weak links
        cursor = await self._conn.execute("""
            SELECT COUNT(*) FROM memory_links WHERE strength < 0.1
        """)
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def prune_weak(self, threshold: float = 0.05) -> int:
        """Remove links below threshold."""
        cursor = await self._conn.execute("""
            DELETE FROM memory_links WHERE strength < ?
        """, (threshold,))
        await self._conn.commit()
        return cursor.rowcount

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAVERSAL
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_outgoing(
        self,
        episode_id: int,
        link_type: Optional[LinkType] = None,
        min_strength: float = 0.0,
    ) -> List[MemoryLink]:
        """Get all links going out from an episode."""
        import json

        if link_type:
            cursor = await self._conn.execute("""
                SELECT * FROM memory_links
                WHERE source_id = ? AND link_type = ? AND strength >= ?
                ORDER BY strength DESC
            """, (episode_id, link_type, min_strength))
        else:
            cursor = await self._conn.execute("""
                SELECT * FROM memory_links
                WHERE source_id = ? AND strength >= ?
                ORDER BY strength DESC
            """, (episode_id, min_strength))

        rows = await cursor.fetchall()
        return [
            MemoryLink(
                source_id=r[1],
                target_id=r[2],
                link_type=LinkType(r[3]),
                strength=r[4],
                created_at=r[5],
                metadata=json.loads(r[6]) if r[6] else None,
            )
            for r in rows
        ]

    async def get_incoming(
        self,
        episode_id: int,
        link_type: Optional[LinkType] = None,
        min_strength: float = 0.0,
    ) -> List[MemoryLink]:
        """Get all links coming into an episode."""
        import json

        if link_type:
            cursor = await self._conn.execute("""
                SELECT * FROM memory_links
                WHERE target_id = ? AND link_type = ? AND strength >= ?
                ORDER BY strength DESC
            """, (episode_id, link_type, min_strength))
        else:
            cursor = await self._conn.execute("""
                SELECT * FROM memory_links
                WHERE target_id = ? AND strength >= ?
                ORDER BY strength DESC
            """, (episode_id, min_strength))

        rows = await cursor.fetchall()
        return [
            MemoryLink(
                source_id=r[1],
                target_id=r[2],
                link_type=LinkType(r[3]),
                strength=r[4],
                created_at=r[5],
                metadata=json.loads(r[6]) if r[6] else None,
            )
            for r in rows
        ]

    async def get_neighbors(
        self,
        episode_id: int,
        min_strength: float = 0.0,
    ) -> Set[int]:
        """Get all episodes connected to this one."""
        outgoing = await self.get_outgoing(episode_id, min_strength=min_strength)
        incoming = await self.get_incoming(episode_id, min_strength=min_strength)

        neighbors = {link.target_id for link in outgoing}
        neighbors.update(link.source_id for link in incoming)
        neighbors.discard(episode_id)  # Remove self if present

        return neighbors

    async def get_node(self, episode_id: int) -> GraphNode:
        """Get full node with all connections."""
        outgoing = await self.get_outgoing(episode_id)
        incoming = await self.get_incoming(episode_id)
        return GraphNode(episode_id, outgoing, incoming)

    # ═══════════════════════════════════════════════════════════════════════════
    # PATH FINDING
    # ═══════════════════════════════════════════════════════════════════════════

    async def find_path(
        self,
        start_id: int,
        end_id: int,
        max_depth: int = 5,
        min_strength: float = 0.1,
    ) -> Optional[List[int]]:
        """
        Find shortest path between two episodes.

        Uses BFS with strength threshold.
        Returns list of episode IDs forming the path, or None.
        """
        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            neighbors = await self.get_neighbors(current, min_strength=min_strength)

            for neighbor in neighbors:
                if neighbor == end_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    async def get_connected_component(
        self,
        episode_id: int,
        max_size: int = 100,
        min_strength: float = 0.1,
    ) -> Set[int]:
        """Get all episodes in the same connected component."""
        visited = set()
        queue = [episode_id]

        while queue and len(visited) < max_size:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            neighbors = await self.get_neighbors(current, min_strength=min_strength)

            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-LINKING
    # ═══════════════════════════════════════════════════════════════════════════

    async def auto_link_temporal(
        self,
        episodes: List[Dict[str, Any]],
        time_threshold: float = 3600.0,  # 1 hour
        strength: float = 0.6,
    ) -> int:
        """
        Auto-link episodes that occurred close in time.

        Episodes within time_threshold seconds get CONTINUES links.
        """
        if len(episodes) < 2:
            return 0

        # Sort by timestamp
        sorted_eps = sorted(episodes, key=lambda e: e.get('created_at', 0))
        links_created = 0

        for i in range(len(sorted_eps) - 1):
            ep1 = sorted_eps[i]
            ep2 = sorted_eps[i + 1]

            t1 = ep1.get('created_at', 0)
            t2 = ep2.get('created_at', 0)

            if t2 - t1 <= time_threshold:
                await self.link(
                    ep1['id'], ep2['id'],
                    LinkType.CONTINUES,
                    strength=strength,
                )
                links_created += 1

        return links_created

    async def auto_link_by_pattern(
        self,
        episodes: List[Dict[str, Any]],
        strength: float = 0.7,
    ) -> int:
        """
        Auto-link episodes with the same Locus trigger pattern.

        Creates RESONATES links.
        """
        # Group by pattern
        by_pattern: Dict[int, List[Dict]] = {}
        for ep in episodes:
            pattern = ep.get('trigger_pattern', 0)
            if pattern not in by_pattern:
                by_pattern[pattern] = []
            by_pattern[pattern].append(ep)

        links_created = 0

        for pattern, eps in by_pattern.items():
            if pattern == 0 or len(eps) < 2:  # Skip NONE pattern
                continue

            # Link each pair (limit to avoid O(n²) explosion)
            for i, ep1 in enumerate(eps[:20]):
                for ep2 in eps[i+1:21]:
                    await self.link(
                        ep1['id'], ep2['id'],
                        LinkType.RESONATES,
                        strength=strength,
                    )
                    links_created += 1

        return links_created

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        # Total links
        cursor = await self._conn.execute("SELECT COUNT(*) FROM memory_links")
        total = (await cursor.fetchone())[0]

        # By type
        cursor = await self._conn.execute("""
            SELECT link_type, COUNT(*) FROM memory_links GROUP BY link_type
        """)
        by_type = {LinkType(r[0]).name: r[1] for r in await cursor.fetchall()}

        # Avg strength
        cursor = await self._conn.execute("SELECT AVG(strength) FROM memory_links")
        avg_strength = (await cursor.fetchone())[0] or 0

        # Unique nodes
        cursor = await self._conn.execute("""
            SELECT COUNT(DISTINCT source_id) + COUNT(DISTINCT target_id) FROM memory_links
        """)
        nodes = (await cursor.fetchone())[0] or 0

        return {
            'total_links': total,
            'by_type': by_type,
            'avg_strength': round(avg_strength, 3),
            'unique_nodes': nodes,
        }

    async def get_most_connected(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """Get episodes with most connections."""
        cursor = await self._conn.execute("""
            SELECT episode_id, COUNT(*) as degree FROM (
                SELECT source_id as episode_id FROM memory_links
                UNION ALL
                SELECT target_id as episode_id FROM memory_links
            )
            GROUP BY episode_id
            ORDER BY degree DESC
            LIMIT ?
        """, (top_k,))

        return [(r[0], r[1]) for r in await cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_graph_memory():
    """Test graph memory operations."""
    print("\n" + "=" * 60)
    print("GRAPH MEMORY TEST")
    print("=" * 60)

    async with GraphMemory('/tmp/test_graph.db') as graph:

        # Create some links
        print("\nCreating links...")
        await graph.link(1, 2, LinkType.CONTINUES, 0.9)
        await graph.link(2, 3, LinkType.CONTINUES, 0.8)
        await graph.link(1, 3, LinkType.REMINDS_OF, 0.7)
        await graph.link(3, 4, LinkType.RESONATES, 0.6)
        await graph.link(4, 5, LinkType.CAUSED_BY, 0.5)
        await graph.link(2, 5, LinkType.CONTRADICTS, 0.4)
        print("  Created 6 links")

        # Test traversal
        print("\nTesting traversal...")
        neighbors = await graph.get_neighbors(2)
        print(f"  Neighbors of 2: {neighbors}")
        assert 1 in neighbors and 3 in neighbors and 5 in neighbors

        # Test path finding
        print("\nTesting path finding...")
        path = await graph.find_path(1, 5)
        print(f"  Path from 1 to 5: {path}")
        assert path is not None
        assert path[0] == 1 and path[-1] == 5

        # Test connected component
        print("\nTesting connected component...")
        component = await graph.get_connected_component(1)
        print(f"  Component from 1: {component}")
        assert len(component) == 5

        # Test strengthen/weaken
        print("\nTesting strength modification...")
        await graph.strengthen(1, 2, 0.1)
        links = await graph.get_outgoing(1)
        continues_link = [l for l in links if l.link_type == LinkType.CONTINUES][0]
        print(f"  Strengthened 1->2: {continues_link.strength}")
        assert continues_link.strength == 1.0  # 0.9 + 0.1 = 1.0

        # Test decay
        print("\nTesting decay...")
        weak_count = await graph.decay_all(0.5)
        print(f"  Links below 0.1 after decay: {weak_count}")

        # Test auto-linking
        print("\nTesting auto-linking...")
        episodes = [
            {'id': 10, 'created_at': 1000, 'trigger_pattern': 1},
            {'id': 11, 'created_at': 1500, 'trigger_pattern': 1},
            {'id': 12, 'created_at': 2000, 'trigger_pattern': 2},
            {'id': 13, 'created_at': 2500, 'trigger_pattern': 1},
        ]

        temporal_links = await graph.auto_link_temporal(episodes, time_threshold=600)
        print(f"  Temporal links created: {temporal_links}")

        pattern_links = await graph.auto_link_by_pattern(episodes)
        print(f"  Pattern links created: {pattern_links}")

        # Stats
        print("\nGraph stats:")
        stats = await graph.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Most connected
        print("\nMost connected nodes:")
        top = await graph.get_most_connected(5)
        for ep_id, degree in top:
            print(f"  Episode {ep_id}: {degree} connections")

    # Cleanup
    import os
    os.remove('/tmp/test_graph.db')

    print("\n" + "=" * 60)
    print("✅ GRAPH MEMORY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_graph_memory())
