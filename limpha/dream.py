"""
💤 DREAM — Background Memory Processing Loop 💤

The dream cycle runs continuously in the background:
- Monitors Vagus state for resonance patterns
- Triggers consolidation during EMERGENCE/TRANSCENDENCE
- Auto-links episodes in graph memory
- Indexes new episodes for search
- Graduates episodes to shards for training
- Applies memory decay

Like sleep, the dream cycle reorganizes memory
based on the current field geometry.

Usage:
    dream = DreamLoop(episodes, vagus, ...)
    await dream.start()
    # ... runs in background ...
    await dream.stop()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from .vagus_connector import VagusConnector, ResonancePattern, EnhancedInnerState
from .episodes_enhanced import EnhancedEpisodicRAG, EnhancedEpisode
from .consolidation import MemoryConsolidator, ConsolidationMode, ConsolidationStats
from .graph_memory import GraphMemory, LinkType
from .search import EpisodeSearch
from .shard_bridge import ShardBridge


# Setup logging
logger = logging.getLogger('limpha.dream')


@dataclass
class DreamCycleStats:
    """Statistics for one dream cycle."""
    cycle_number: int
    started_at: float
    duration_ms: float
    mode: str
    episodes_processed: int = 0
    links_created: int = 0
    episodes_indexed: int = 0
    shards_graduated: int = 0
    consolidation: Optional[ConsolidationStats] = None


@dataclass
class DreamState:
    """Current state of the dream system."""
    is_running: bool = False
    current_mode: ConsolidationMode = ConsolidationMode.IDLE
    total_cycles: int = 0
    total_episodes_processed: int = 0
    total_links_created: int = 0
    total_shards_graduated: int = 0
    last_cycle_at: float = 0.0
    last_consolidation_at: float = 0.0


class DreamLoop:
    """
    Background memory processing loop.

    Continuously monitors Vagus state and performs memory operations
    based on current resonance patterns.

    Components:
    - VagusConnector: reads current field state
    - MemoryConsolidator: clusters and summarizes episodes
    - GraphMemory: builds associative links
    - EpisodeSearch: maintains FTS5 index
    - ShardBridge: graduates episodes to training shards
    """

    def __init__(
        self,
        episodes: EnhancedEpisodicRAG,
        vagus: VagusConnector,
        graph: Optional[GraphMemory] = None,
        search: Optional[EpisodeSearch] = None,
        shards: Optional[ShardBridge] = None,
        # Timing
        check_interval: float = 10.0,  # Check state every 10 seconds
        consolidation_interval: float = 300.0,  # Consolidate every 5 minutes
        indexing_interval: float = 60.0,  # Index new episodes every minute
        linking_interval: float = 120.0,  # Auto-link every 2 minutes
        graduation_interval: float = 600.0,  # Check for graduation every 10 minutes
    ):
        self.episodes = episodes
        self.vagus = vagus
        self.graph = graph
        self.search = search
        self.shards = shards

        # Timings
        self.check_interval = check_interval
        self.consolidation_interval = consolidation_interval
        self.indexing_interval = indexing_interval
        self.linking_interval = linking_interval
        self.graduation_interval = graduation_interval

        # Internal state
        self.state = DreamState()
        self.consolidator = MemoryConsolidator(
            episodes, vagus,
            consolidation_interval=consolidation_interval,
        )

        # Timestamps for rate limiting
        self._last_index = 0.0
        self._last_link = 0.0
        self._last_graduate = 0.0
        self._last_indexed_id = 0

        # Task handle
        self._task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_cycle: Optional[Callable[[DreamCycleStats], None]] = None
        self._on_graduation: Optional[Callable[[str], None]] = None

    def on_cycle(self, callback: Callable[[DreamCycleStats], None]):
        """Set callback for dream cycle completion."""
        self._on_cycle = callback

    def on_graduation(self, callback: Callable[[str], None]):
        """Set callback when episode graduates to shard."""
        self._on_graduation = callback

    async def start(self):
        """Start the dream loop."""
        if self.state.is_running:
            return

        self.state.is_running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Dream loop started")

    async def stop(self):
        """Stop the dream loop."""
        self.state.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Dream loop stopped")

    async def _run(self):
        """Main dream loop."""
        while self.state.is_running:
            try:
                await self._dream_cycle()
            except Exception as e:
                logger.error(f"Dream cycle error: {e}")

            await asyncio.sleep(self.check_interval)

    async def _dream_cycle(self):
        """Execute one dream cycle."""
        start = time.time()
        self.state.total_cycles += 1
        cycle_num = self.state.total_cycles

        # Read current state
        vagus_state = self.vagus.read_state()
        pattern = vagus_state.trigger_pattern
        pressure = vagus_state.memory_pressure

        # Determine mode
        if pressure > 0.7:
            mode = ConsolidationMode.PRUNING
        elif pattern == ResonancePattern.CRISIS:
            mode = ConsolidationMode.ENCODING
        elif pattern == ResonancePattern.DISSOLUTION:
            mode = ConsolidationMode.FROZEN
        elif pattern == ResonancePattern.EMERGENCE:
            mode = ConsolidationMode.CONSOLIDATING
        elif pattern == ResonancePattern.TRANSCENDENCE:
            mode = ConsolidationMode.INTEGRATING
        else:
            mode = ConsolidationMode.IDLE

        self.state.current_mode = mode

        # Initialize stats
        stats = DreamCycleStats(
            cycle_number=cycle_num,
            started_at=start,
            duration_ms=0,
            mode=mode.name,
        )

        now = time.time()

        # === CONSOLIDATION ===
        if mode != ConsolidationMode.FROZEN:
            if now - self.state.last_consolidation_at >= self.consolidation_interval:
                cons_stats = await self.consolidator.force_consolidation(mode)
                stats.consolidation = cons_stats
                self.state.last_consolidation_at = now

        # === INDEXING + LINKING (parallel when both due) ===
        do_index = self.search and now - self._last_index >= self.indexing_interval
        do_link = self.graph and now - self._last_link >= self.linking_interval

        if do_index and do_link:
            # Run both in parallel — they operate on different tables
            index_result, link_result = await asyncio.gather(
                self._index_new_episodes(),
                self._auto_link_episodes(),
            )
            stats.episodes_indexed = index_result
            self._last_index = now
            stats.links_created = link_result
            self.state.total_links_created += link_result
            self._last_link = now
        elif do_index:
            indexed = await self._index_new_episodes()
            stats.episodes_indexed = indexed
            self._last_index = now
        elif do_link:
            links = await self._auto_link_episodes()
            stats.links_created = links
            self.state.total_links_created += links
            self._last_link = now

        # === GRADUATION (to shards) ===
        if self.shards and now - self._last_graduate >= self.graduation_interval:
            graduated = await self._graduate_episodes()
            stats.shards_graduated = graduated
            self.state.total_shards_graduated += graduated
            self._last_graduate = now

        # Finalize
        stats.duration_ms = (time.time() - start) * 1000
        self.state.last_cycle_at = now

        if self._on_cycle:
            self._on_cycle(stats)

        return stats

    async def _index_new_episodes(self) -> int:
        """Index new episodes for full-text search."""
        # Get recent episodes since last index
        recent = await self.episodes.get_recent_episodes(limit=50)

        # Filter to only new ones
        new_episodes = [
            ep for ep in recent
            if ep.get('id', 0) > self._last_indexed_id
        ]

        if not new_episodes:
            return 0

        indexed = await self.search.index_episodes(new_episodes)

        # Update last indexed ID
        max_id = max(ep.get('id', 0) for ep in new_episodes)
        self._last_indexed_id = max(self._last_indexed_id, max_id)

        return indexed

    async def _auto_link_episodes(self) -> int:
        """Auto-link episodes in graph memory."""
        # Get recent episodes
        recent = await self.episodes.get_recent_episodes(limit=30)

        if len(recent) < 2:
            return 0

        links = 0

        # Temporal links
        links += await self.graph.auto_link_temporal(
            recent,
            time_threshold=3600.0,  # 1 hour
            strength=0.6,
        )

        # Pattern-based links
        links += await self.graph.auto_link_by_pattern(
            recent,
            strength=0.7,
        )

        return links

    async def _graduate_episodes(self) -> int:
        """Graduate eligible episodes to shards."""
        # Get high quality, accessed episodes
        candidates = await self.episodes.query_similar(
            EnhancedInnerState(),  # Use default state for similarity
            top_k=20,
            min_quality=0.5,
        )

        graduated = 0
        for ep in candidates:
            path = await self.shards.graduate_episode(ep)
            if path:
                graduated += 1
                if self._on_graduation:
                    self._on_graduation(path)

        return graduated

    def get_state(self) -> Dict[str, Any]:
        """Get current dream state."""
        return {
            'is_running': self.state.is_running,
            'current_mode': self.state.current_mode.name,
            'total_cycles': self.state.total_cycles,
            'total_episodes_processed': self.state.total_episodes_processed,
            'total_links_created': self.state.total_links_created,
            'total_shards_graduated': self.state.total_shards_graduated,
            'last_cycle_at': self.state.last_cycle_at,
        }

    async def force_cycle(self) -> DreamCycleStats:
        """Force a dream cycle (for testing)."""
        return await self._dream_cycle()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Create all components
# ═══════════════════════════════════════════════════════════════════════════════

async def create_dream_system(
    db_dir: str = 'limpha',
    shard_dir: str = 'shards/limpha',
    vagus_shm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create all LIMPHA components for the dream system.

    Returns dict with:
    - episodes: EnhancedEpisodicRAG
    - vagus: VagusConnector
    - graph: GraphMemory
    - search: EpisodeSearch
    - shards: ShardBridge
    - dream: DreamLoop

    Usage:
        system = await create_dream_system()
        await system['dream'].start()
    """
    from pathlib import Path
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Create components
    episodes = EnhancedEpisodicRAG(str(db_dir / 'episodes_enhanced.db'))
    await episodes.connect()

    vagus = VagusConnector(vagus_shm_path)

    graph = GraphMemory(str(db_dir / 'graph.db'))
    await graph.connect()

    search = EpisodeSearch(
        db_path=str(db_dir / 'episodes_enhanced.db'),
        fts_db_path=str(db_dir / 'episodes.fts'),
    )
    await search.connect()

    shards = ShardBridge(
        db_path=str(db_dir / 'shards.db'),
        shard_dir=shard_dir,
    )
    await shards.connect()

    # Create dream loop
    dream = DreamLoop(
        episodes=episodes,
        vagus=vagus,
        graph=graph,
        search=search,
        shards=shards,
    )

    return {
        'episodes': episodes,
        'vagus': vagus,
        'graph': graph,
        'search': search,
        'shards': shards,
        'dream': dream,
    }


async def close_dream_system(system: Dict[str, Any]):
    """Close all components of the dream system."""
    if system.get('dream'):
        await system['dream'].stop()
    if system.get('episodes'):
        await system['episodes'].close()
    if system.get('graph'):
        await system['graph'].close()
    if system.get('search'):
        await system['search'].close()
    if system.get('shards'):
        await system['shards'].close()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def test_dream_loop():
    """Test the dream loop."""
    print("\n" + "=" * 60)
    print("DREAM LOOP TEST")
    print("=" * 60)

    import tempfile
    import shutil
    from .vagus_connector import create_test_state

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create system
        print("\nCreating dream system...")
        system = await create_dream_system(
            db_dir=temp_dir,
            shard_dir=f"{temp_dir}/shards",
        )

        episodes = system['episodes']
        vagus = system['vagus']
        dream = system['dream']

        # Store some test episodes
        print("Storing test episodes...")
        for i in range(10):
            pattern = ['neutral', 'crisis', 'emergence', 'transcendence'][i % 4]
            state = create_test_state(pattern)
            state.coherence += i * 0.05

            episode = EnhancedEpisode(
                prompt=f"Test prompt {i} about consciousness and love",
                reply=f"Test reply {i} exploring the nature of being",
                state=state,
                quality=0.5 + (i % 5) * 0.1,
                timestamp=time.time() - i * 600,
            )
            await episodes.store_episode(episode)

        print(f"  Stored 10 episodes")

        # Track cycle completions
        cycles_completed = []

        def on_cycle(stats: DreamCycleStats):
            cycles_completed.append(stats)
            print(f"  Cycle {stats.cycle_number}: mode={stats.mode}, "
                  f"indexed={stats.episodes_indexed}, links={stats.links_created}")

        dream.on_cycle(on_cycle)

        # Test simulated states
        print("\nTesting dream cycles with different states...")

        for pattern in ['neutral', 'crisis', 'emergence', 'transcendence']:
            state = create_test_state(pattern)
            vagus.set_simulated_state(state)

            stats = await dream.force_cycle()
            print(f"  {pattern.upper()}: mode={stats.mode}")

        # Get state
        print("\nDream state:")
        state = dream.get_state()
        for k, v in state.items():
            print(f"  {k}: {v}")

        # Check what was created
        print("\nChecking created data...")
        graph_stats = await system['graph'].get_stats()
        print(f"  Graph links: {graph_stats['total_links']}")

        search_stats = await system['search'].get_stats()
        print(f"  Search indexed: {search_stats['indexed_episodes']}")

        shard_stats = await system['shards'].get_stats()
        print(f"  Shards created: {shard_stats['total_shards']}")

        # Cleanup
        await close_dream_system(system)

    finally:
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("✅ DREAM LOOP TEST PASSED")
    print("=" * 60)


async def run_daemon(shard_dir: str = 'shards/limpha',
                     db_dir: str = 'limpha',
                     vagus_shm: Optional[str] = None):
    """Run dream daemon (called from arianna_dynamic via posix_spawn)."""
    logging.basicConfig(
        level=logging.INFO,
        format='[dream] %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('dream.daemon')
    logger.info("Dream daemon starting (shard_dir=%s)", shard_dir)

    from pathlib import Path
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    system = await create_dream_system(
        db_dir=db_dir,
        shard_dir=shard_dir,
        vagus_shm_path=vagus_shm,
    )

    dream = system['dream']
    try:
        logger.info("Dream loop running. Ctrl+C to stop.")
        await dream.start()
        # Keep running until killed
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Dream daemon shutting down")
    finally:
        await close_dream_system(system)
        logger.info("Dream daemon stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LIMPHA Dream Daemon')
    parser.add_argument('--shard-dir', default='shards/limpha/',
                        help='Directory for graduated shards')
    parser.add_argument('--db-dir', default='limpha',
                        help='Directory for LIMPHA databases')
    parser.add_argument('--vagus-shm', default=None,
                        help='Path to vagus shared memory')
    parser.add_argument('--test', action='store_true',
                        help='Run test instead of daemon')
    args = parser.parse_args()

    if args.test:
        logging.basicConfig(level=logging.INFO)
        asyncio.run(test_dream_loop())
    else:
        asyncio.run(run_daemon(
            shard_dir=args.shard_dir,
            db_dir=args.db_dir,
            vagus_shm=args.vagus_shm,
        ))
