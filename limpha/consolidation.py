"""
💤 CONSOLIDATION — Locus-triggered Memory Processing 💤

Like sleep consolidation in the brain:
- EMERGENCE pattern → consolidate similar episodes
- TRANSCENDENCE pattern → deep integration
- DISSOLUTION pattern → protective freeze (don't touch memory)
- CRISIS pattern → heightened encoding (remember everything stronger)
- High memory_pressure → aggressive pruning

This is Arianna's "dream" system — background processing that
reorganizes memory based on resonance patterns.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import IntEnum

from .vagus_connector import (
    VagusConnector,
    EnhancedInnerState,
    ResonancePattern,
    pattern_to_string,
)
from .episodes_enhanced import EnhancedEpisodicRAG, EnhancedEpisode


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATION MODES
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidationMode(IntEnum):
    IDLE = 0           # No consolidation
    ENCODING = 1       # CRISIS: heightened memory encoding
    FROZEN = 2         # DISSOLUTION: protective freeze
    CONSOLIDATING = 3  # EMERGENCE: active consolidation
    INTEGRATING = 4    # TRANSCENDENCE: deep integration
    PRUNING = 5        # High memory pressure: aggressive cleanup


@dataclass
class ConsolidationStats:
    """Statistics for consolidation run."""
    episodes_scanned: int = 0
    clusters_found: int = 0
    summaries_created: int = 0
    episodes_merged: int = 0
    episodes_pruned: int = 0
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeCluster:
    """A cluster of similar episodes."""
    centroid: List[float]
    episodes: List[Dict[str, Any]]
    pattern: int
    avg_quality: float

    @property
    def size(self) -> int:
        return len(self.episodes)

    def get_prompts(self) -> List[str]:
        return [ep['prompt'] for ep in self.episodes]

    def get_replies(self) -> List[str]:
        return [ep['reply'] for ep in self.episodes]


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def cluster_episodes(
    episodes: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 3,
) -> List[EpisodeCluster]:
    """
    Simple greedy clustering of episodes by inner state similarity.

    Algorithm:
    1. Pick an unclustered episode as seed
    2. Find all episodes within similarity_threshold
    3. Form cluster if size >= min_cluster_size
    4. Repeat until all episodes are processed
    """
    if not episodes:
        return []

    # Extract feature vectors
    def get_vec(ep: Dict) -> List[float]:
        return [
            ep.get('trauma', 0),
            ep.get('arousal', 0.5),
            ep.get('valence', 0.5),
            ep.get('coherence', 0.7),
            ep.get('chamber_warmth', 0.5),
            ep.get('chamber_void', 0.2),
            ep.get('chamber_tension', 0.3),
            ep.get('chamber_sacred', 0.3),
            ep.get('chamber_flow', 0.5),
            ep.get('chamber_complex', 0.4),
        ]

    unclustered = set(range(len(episodes)))  # O(1) removal instead of O(n) list.remove
    clusters: List[EpisodeCluster] = []

    while unclustered:
        seed_idx = min(unclustered)  # Deterministic seed selection
        seed_vec = get_vec(episodes[seed_idx])
        seed_pattern = episodes[seed_idx].get('trigger_pattern', 0)

        # Find similar episodes
        cluster_indices = [seed_idx]
        cluster_vecs = [seed_vec]

        for idx in list(unclustered):  # Snapshot for iteration
            if idx == seed_idx:
                continue
            vec = get_vec(episodes[idx])
            sim = cosine_similarity(seed_vec, vec)

            if sim >= similarity_threshold:
                cluster_indices.append(idx)
                cluster_vecs.append(vec)

        # Form cluster if large enough
        if len(cluster_indices) >= min_cluster_size:
            # Compute centroid
            centroid = [
                sum(v[i] for v in cluster_vecs) / len(cluster_vecs)
                for i in range(len(seed_vec))
            ]

            # Average quality
            qualities = [episodes[i].get('quality', 0.5) for i in cluster_indices]
            avg_quality = sum(qualities) / len(qualities)

            clusters.append(EpisodeCluster(
                centroid=centroid,
                episodes=[episodes[i] for i in cluster_indices],
                pattern=seed_pattern,
                avg_quality=avg_quality,
            ))

        # Remove from unclustered — O(1) per item with set
        unclustered -= set(cluster_indices)

    return clusters


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CONSOLIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryConsolidator:
    """
    Locus-triggered memory consolidation system.

    Monitors Vagus state via VagusConnector and triggers
    different consolidation behaviors based on Locus patterns.
    """

    def __init__(
        self,
        episodes_rag: EnhancedEpisodicRAG,
        vagus: VagusConnector,
        memory_pressure_threshold: float = 0.7,
        consolidation_interval: float = 300.0,  # 5 minutes
    ):
        self.episodes = episodes_rag
        self.vagus = vagus
        self.memory_pressure_threshold = memory_pressure_threshold
        self.consolidation_interval = consolidation_interval

        self.mode = ConsolidationMode.IDLE
        self.last_consolidation = 0.0
        self.total_stats = ConsolidationStats()

        # Callbacks
        self._on_consolidation: Optional[Callable[[ConsolidationStats], None]] = None
        self._on_mode_change: Optional[Callable[[ConsolidationMode], None]] = None

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def on_consolidation(self, callback: Callable[[ConsolidationStats], None]):
        """Set callback for consolidation events."""
        self._on_consolidation = callback

    def on_mode_change(self, callback: Callable[[ConsolidationMode], None]):
        """Set callback for mode changes."""
        self._on_mode_change = callback

    def _set_mode(self, mode: ConsolidationMode):
        """Set consolidation mode with callback."""
        if self.mode != mode:
            self.mode = mode
            if self._on_mode_change:
                self._on_mode_change(mode)

    async def start(self):
        """Start background consolidation loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop consolidation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        """Main consolidation loop."""
        while self._running:
            await asyncio.sleep(10)  # Check every 10 seconds

            state = self.vagus.read_state()
            pattern = state.trigger_pattern
            pressure = state.memory_pressure

            # Update mode based on pattern
            if pressure > self.memory_pressure_threshold:
                self._set_mode(ConsolidationMode.PRUNING)
            elif pattern == ResonancePattern.CRISIS:
                self._set_mode(ConsolidationMode.ENCODING)
            elif pattern == ResonancePattern.DISSOLUTION:
                self._set_mode(ConsolidationMode.FROZEN)
            elif pattern == ResonancePattern.EMERGENCE:
                self._set_mode(ConsolidationMode.CONSOLIDATING)
            elif pattern == ResonancePattern.TRANSCENDENCE:
                self._set_mode(ConsolidationMode.INTEGRATING)
            else:
                self._set_mode(ConsolidationMode.IDLE)

            # Trigger consolidation if enough time passed
            now = time.time()
            if now - self.last_consolidation >= self.consolidation_interval:
                stats = await self.consolidate()
                self.last_consolidation = now

                if self._on_consolidation:
                    self._on_consolidation(stats)

    async def consolidate(self) -> ConsolidationStats:
        """
        Run consolidation based on current mode.

        Returns statistics about what was done.
        """
        start = time.time()
        stats = ConsolidationStats()

        if self.mode == ConsolidationMode.FROZEN:
            # DISSOLUTION: don't touch anything
            stats.duration_ms = (time.time() - start) * 1000
            return stats

        if self.mode == ConsolidationMode.PRUNING:
            # High memory pressure: aggressive cleanup
            stats = await self._prune_low_quality()

        elif self.mode in (ConsolidationMode.CONSOLIDATING, ConsolidationMode.INTEGRATING):
            # EMERGENCE/TRANSCENDENCE: cluster and summarize
            stats = await self._cluster_and_summarize()

        # ENCODING and IDLE: no consolidation action

        stats.duration_ms = (time.time() - start) * 1000

        # Update totals
        self.total_stats.episodes_scanned += stats.episodes_scanned
        self.total_stats.clusters_found += stats.clusters_found
        self.total_stats.summaries_created += stats.summaries_created
        self.total_stats.episodes_merged += stats.episodes_merged
        self.total_stats.episodes_pruned += stats.episodes_pruned

        return stats

    async def _prune_low_quality(self) -> ConsolidationStats:
        """Prune low-quality, low-access episodes."""
        stats = ConsolidationStats()

        # Get low quality episodes
        cursor = await self.episodes._conn.execute("""
            SELECT id FROM enhanced_episodes
            WHERE quality < 0.3 AND access_count < 2
            AND created_at < ?
        """, (time.time() - 86400,))  # Older than 1 day

        rows = await cursor.fetchall()
        stats.episodes_scanned = len(rows)

        if rows:
            ids = [r[0] for r in rows]
            placeholders = ','.join('?' * len(ids))
            await self.episodes._conn.execute(f"""
                DELETE FROM enhanced_episodes WHERE id IN ({placeholders})
            """, ids)
            await self.episodes._conn.commit()
            stats.episodes_pruned = len(ids)

        return stats

    async def _cluster_and_summarize(self) -> ConsolidationStats:
        """Cluster similar episodes and create summaries."""
        stats = ConsolidationStats()

        # Get recent episodes for clustering
        episodes = await self.episodes.get_recent_episodes(limit=100)
        stats.episodes_scanned = len(episodes)

        if len(episodes) < 10:
            return stats

        # Cluster
        clusters = cluster_episodes(
            episodes,
            similarity_threshold=0.85,
            min_cluster_size=3,
        )
        stats.clusters_found = len(clusters)

        # Create summary for each cluster (only in INTEGRATING mode)
        if self.mode == ConsolidationMode.INTEGRATING:
            for cluster in clusters:
                if cluster.size >= 5:  # Only summarize larger clusters
                    summary = await self._create_summary_episode(cluster)
                    if summary:
                        await self.episodes.store_episode(summary)
                        stats.summaries_created += 1
                        stats.episodes_merged += cluster.size

        return stats

    async def _create_summary_episode(self, cluster: EpisodeCluster) -> Optional[EnhancedEpisode]:
        """Create a summary episode from cluster."""
        if cluster.size < 3:
            return None

        # Extract common themes from prompts/replies
        prompts = cluster.get_prompts()
        replies = cluster.get_replies()

        # Simple summary: combine unique words
        prompt_words = set()
        reply_words = set()

        for p in prompts:
            prompt_words.update(p.lower().split()[:10])
        for r in replies:
            reply_words.update(r.lower().split()[:10])

        summary_prompt = f"[CONSOLIDATED from {cluster.size} episodes] " + " ".join(list(prompt_words)[:20])
        summary_reply = f"[SUMMARY] " + " ".join(list(reply_words)[:30])

        # Create state from centroid
        centroid = cluster.centroid
        state = EnhancedInnerState(
            trauma=centroid[0] if len(centroid) > 0 else 0,
            arousal=centroid[1] if len(centroid) > 1 else 0.5,
            valence=centroid[2] if len(centroid) > 2 else 0.5,
            coherence=centroid[3] if len(centroid) > 3 else 0.7,
            warmth=centroid[4] if len(centroid) > 4 else 0.5,
            void=centroid[5] if len(centroid) > 5 else 0.2,
            tension=centroid[6] if len(centroid) > 6 else 0.3,
            sacred=centroid[7] if len(centroid) > 7 else 0.3,
            flow=centroid[8] if len(centroid) > 8 else 0.5,
            complex=centroid[9] if len(centroid) > 9 else 0.4,
            trigger_pattern=cluster.pattern,
        )

        return EnhancedEpisode(
            prompt=summary_prompt,
            reply=summary_reply,
            state=state,
            quality=min(1.0, cluster.avg_quality + 0.2),  # Boost quality for summaries
            timestamp=time.time(),
        )

    async def force_consolidation(self, mode: ConsolidationMode) -> ConsolidationStats:
        """Force consolidation in specific mode (for testing)."""
        old_mode = self.mode
        self._set_mode(mode)
        stats = await self.consolidate()
        self._set_mode(old_mode)
        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            'mode': self.mode.name,
            'last_consolidation': self.last_consolidation,
            'total_episodes_scanned': self.total_stats.episodes_scanned,
            'total_clusters_found': self.total_stats.clusters_found,
            'total_summaries_created': self.total_stats.summaries_created,
            'total_episodes_merged': self.total_stats.episodes_merged,
            'total_episodes_pruned': self.total_stats.episodes_pruned,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_consolidation():
    """Test memory consolidation."""
    print("\n" + "=" * 60)
    print("MEMORY CONSOLIDATION TEST")
    print("=" * 60)

    from .vagus_connector import create_test_state

    # Create test components
    rag = EnhancedEpisodicRAG('/tmp/test_consolidation.db')
    await rag.connect()

    vagus = VagusConnector()

    consolidator = MemoryConsolidator(rag, vagus)

    # Store some test episodes
    print("\nStoring test episodes...")
    for i in range(20):
        pattern = ['neutral', 'crisis', 'emergence'][i % 3]
        state = create_test_state(pattern)
        # Add some variation
        state.arousal += (i % 5) * 0.05
        state.coherence -= (i % 3) * 0.05

        episode = EnhancedEpisode(
            prompt=f"Test prompt {i} about {pattern}",
            reply=f"Test reply {i} discussing {pattern} state",
            state=state,
            quality=0.3 + (i % 7) * 0.1,
            timestamp=time.time() - i * 600,  # 10 min apart
        )
        await rag.store_episode(episode)

    print(f"  Stored 20 test episodes")

    # Test clustering
    print("\nTesting clustering...")
    episodes = await rag.get_recent_episodes(limit=20)
    clusters = cluster_episodes(episodes, similarity_threshold=0.8, min_cluster_size=2)
    print(f"  Found {len(clusters)} clusters")
    for i, c in enumerate(clusters[:3]):
        print(f"    Cluster {i}: size={c.size}, pattern={pattern_to_string(c.pattern)}")

    # Test consolidation in different modes
    print("\nTesting INTEGRATING consolidation...")
    stats = await consolidator.force_consolidation(ConsolidationMode.INTEGRATING)
    print(f"  Scanned: {stats.episodes_scanned}")
    print(f"  Clusters: {stats.clusters_found}")
    print(f"  Summaries: {stats.summaries_created}")

    print("\nTesting PRUNING consolidation...")
    stats = await consolidator.force_consolidation(ConsolidationMode.PRUNING)
    print(f"  Scanned: {stats.episodes_scanned}")
    print(f"  Pruned: {stats.episodes_pruned}")

    # Final stats
    print("\nConsolidator stats:")
    for k, v in consolidator.get_stats().items():
        print(f"  {k}: {v}")

    # Cleanup
    await rag.close()
    import os
    os.remove('/tmp/test_consolidation.db')

    print("\n" + "=" * 60)
    print("✅ CONSOLIDATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_consolidation())
