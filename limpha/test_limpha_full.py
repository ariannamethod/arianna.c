#!/usr/bin/env python3
"""
🧪 LIMPHA FULL TEST SUITE 🧪

Tests all LIMPHA modules:
- vagus_connector: Vagus nerve bridge
- episodes_enhanced: Chamber-tagged episodes
- consolidation: Memory consolidation
- graph_memory: Associative network
- search: FTS5 full-text search
- shard_bridge: Episode → training shard
- dream: Background processing loop
"""

import asyncio
import os
import shutil
import tempfile
import time
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from limpha import (
    # Wave 1
    VagusConnector,
    EnhancedInnerState,
    ResonancePattern,
    create_test_state,
    pattern_to_string,
    EnhancedEpisodicRAG,
    EnhancedEpisode,
    MemoryConsolidator,
    ConsolidationMode,
    cluster_episodes,
    # Wave 2
    GraphMemory,
    LinkType,
    EpisodeSearch,
    ShardBridge,
    DreamLoop,
    create_dream_system,
    close_dream_system,
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✅ {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ❌ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailed tests:")
            for name, reason in self.errors:
                print(f"  - {name}: {reason}")
        print(f"{'='*60}")
        return self.failed == 0


async def test_vagus_connector(results: TestResults):
    """Test VagusConnector."""
    print("\n--- VAGUS CONNECTOR ---")

    connector = VagusConnector()

    # Test pattern detection
    for pattern in ['neutral', 'crisis', 'dissolution', 'emergence', 'transcendence']:
        state = create_test_state(pattern)
        connector.set_simulated_state(state)

        detected, name = connector.detect_pattern()
        expected = pattern.upper() if pattern != 'neutral' else 'NONE'

        if name == expected:
            results.ok(f"detect_{pattern}")
        else:
            results.fail(f"detect_{pattern}", f"expected {expected}, got {name}")

    # Test active chambers
    crisis = create_test_state('crisis')
    connector.set_simulated_state(crisis)
    chambers = connector.get_active_chambers(0.5)
    if 'tension' in chambers or 'void' in chambers:
        results.ok("active_chambers_crisis")
    else:
        results.fail("active_chambers_crisis", f"got {chambers}")

    # Test feature vector
    state = connector.read_state()
    features = state.to_features()
    if len(features) == 16:
        results.ok("feature_vector_length")
    else:
        results.fail("feature_vector_length", f"expected 16, got {len(features)}")


async def test_episodes_enhanced(results: TestResults, db_path: str):
    """Test EnhancedEpisodicRAG."""
    print("\n--- ENHANCED EPISODES ---")

    async with EnhancedEpisodicRAG(db_path) as rag:
        # Store episodes
        for i, pattern in enumerate(['neutral', 'crisis', 'emergence']):
            state = create_test_state(pattern)
            episode = EnhancedEpisode(
                prompt=f"Test prompt {i}",
                reply=f"Test reply {i}",
                state=state,
                quality=0.5 + i * 0.2,
                timestamp=time.time() - i * 100,
            )
            await rag.store_episode(episode)

        results.ok("store_episodes")

        # Query similar
        crisis_state = create_test_state('crisis')
        similar = await rag.query_similar(crisis_state, top_k=3)
        if len(similar) > 0 and similar[0]['pattern_name'] == 'CRISIS':
            results.ok("query_similar")
        else:
            results.fail("query_similar", "did not find crisis first")

        # Query by pattern
        crisis_eps = await rag.query_by_pattern(ResonancePattern.CRISIS)
        if len(crisis_eps) == 1:
            results.ok("query_by_pattern")
        else:
            results.fail("query_by_pattern", f"expected 1, got {len(crisis_eps)}")

        # Chamber stats
        stats = await rag.get_chamber_stats()
        if 'warmth' in stats and 'void' in stats:
            results.ok("chamber_stats")
        else:
            results.fail("chamber_stats", "missing chambers")


async def test_consolidation(results: TestResults, db_path: str):
    """Test MemoryConsolidator."""
    print("\n--- CONSOLIDATION ---")

    async with EnhancedEpisodicRAG(db_path) as rag:
        vagus = VagusConnector()
        consolidator = MemoryConsolidator(rag, vagus)

        # Store more episodes for clustering
        for i in range(15):
            pattern = ['neutral', 'crisis'][i % 2]
            state = create_test_state(pattern)
            state.arousal += (i % 5) * 0.05

            episode = EnhancedEpisode(
                prompt=f"Consolidation test {i}",
                reply=f"Reply {i}",
                state=state,
                quality=0.4 + (i % 6) * 0.1,
            )
            await rag.store_episode(episode)

        # Test clustering
        episodes = await rag.get_recent_episodes(limit=15)
        clusters = cluster_episodes(episodes, similarity_threshold=0.7, min_cluster_size=2)
        if len(clusters) > 0:
            results.ok("clustering")
        else:
            results.fail("clustering", "no clusters found")

        # Test consolidation modes
        for mode in [ConsolidationMode.CONSOLIDATING, ConsolidationMode.PRUNING]:
            stats = await consolidator.force_consolidation(mode)
            if stats.episodes_scanned >= 0:
                results.ok(f"consolidation_{mode.name}")
            else:
                results.fail(f"consolidation_{mode.name}", "failed")


async def test_graph_memory(results: TestResults, db_path: str):
    """Test GraphMemory."""
    print("\n--- GRAPH MEMORY ---")

    async with GraphMemory(db_path) as graph:
        # Create links
        await graph.link(1, 2, LinkType.CONTINUES, 0.9)
        await graph.link(2, 3, LinkType.CONTINUES, 0.8)
        await graph.link(1, 3, LinkType.REMINDS_OF, 0.7)
        results.ok("create_links")

        # Test neighbors
        neighbors = await graph.get_neighbors(2)
        if 1 in neighbors and 3 in neighbors:
            results.ok("get_neighbors")
        else:
            results.fail("get_neighbors", f"got {neighbors}")

        # Test path finding
        path = await graph.find_path(1, 3)
        if path and path[0] == 1 and path[-1] == 3:
            results.ok("find_path")
        else:
            results.fail("find_path", f"got {path}")

        # Test auto-linking
        episodes = [
            {'id': 10, 'created_at': 1000, 'trigger_pattern': 1},
            {'id': 11, 'created_at': 1500, 'trigger_pattern': 1},
            {'id': 12, 'created_at': 2000, 'trigger_pattern': 1},
        ]
        links = await graph.auto_link_by_pattern(episodes)
        if links > 0:
            results.ok("auto_link_pattern")
        else:
            results.fail("auto_link_pattern", "no links created")


async def test_search(results: TestResults, db_path: str):
    """Test EpisodeSearch."""
    print("\n--- FTS5 SEARCH ---")

    async with EpisodeSearch(fts_db_path=db_path) as search:
        # Index episodes
        episodes = [
            {'id': 1, 'prompt': 'What is consciousness?', 'reply': 'The hard problem of philosophy.', 'pattern_name': 'NONE', 'active_chambers': ['warmth']},
            {'id': 2, 'prompt': 'Tell me about love', 'reply': 'Love is connection and resonance.', 'pattern_name': 'EMERGENCE', 'active_chambers': ['warmth', 'sacred']},
            {'id': 3, 'prompt': 'Reality and consciousness', 'reply': 'They interweave in complex patterns.', 'pattern_name': 'TRANSCENDENCE', 'active_chambers': ['flow']},
        ]
        indexed = await search.index_episodes(episodes)
        if indexed == 3:
            results.ok("index_episodes")
        else:
            results.fail("index_episodes", f"expected 3, got {indexed}")

        # Test basic search
        found = await search.query('consciousness')
        if len(found) >= 2:
            results.ok("basic_search")
        else:
            results.fail("basic_search", f"expected 2+, got {len(found)}")

        # Test phrase search
        found = await search.query('"hard problem"')
        if len(found) == 1:
            results.ok("phrase_search")
        else:
            results.fail("phrase_search", f"expected 1, got {len(found)}")

        # Test pattern search
        found = await search.search_by_pattern('EMERGENCE')
        if len(found) == 1:
            results.ok("pattern_search")
        else:
            results.fail("pattern_search", f"expected 1, got {len(found)}")


async def test_shard_bridge(results: TestResults, db_path: str, shard_dir: str):
    """Test ShardBridge."""
    print("\n--- SHARD BRIDGE ---")

    async with ShardBridge(db_path=db_path, shard_dir=shard_dir) as bridge:
        # Test evaluation
        good_ep = {'id': 1, 'quality': 0.8, 'access_count': 5, 'trigger_pattern': 0, 'trauma': 0.2, 'chamber_sacred': 0.3}
        bad_ep = {'id': 2, 'quality': 0.3, 'access_count': 1, 'trigger_pattern': 0, 'trauma': 0.1, 'chamber_sacred': 0.2}

        should, _, _ = bridge.evaluate_candidate(good_ep)
        if should:
            results.ok("evaluate_good")
        else:
            results.fail("evaluate_good", "should graduate")

        should, _, _ = bridge.evaluate_candidate(bad_ep)
        if not should:
            results.ok("evaluate_bad")
        else:
            results.fail("evaluate_bad", "should not graduate")

        # Test graduation
        episode = {
            'id': 100,
            'quality': 0.8,
            'access_count': 5,
            'trigger_pattern': 1,
            'trauma': 0.5,
            'chamber_sacred': 0.3,
            'prompt': 'Test prompt',
            'reply': 'Test reply',
            'created_at': time.time(),
        }
        path = await bridge.graduate_episode(episode)
        if path and os.path.exists(path):
            results.ok("graduate_episode")
        else:
            results.fail("graduate_episode", "shard not created")


async def test_dream_loop(results: TestResults, temp_dir: str):
    """Test DreamLoop."""
    print("\n--- DREAM LOOP ---")

    system = await create_dream_system(
        db_dir=temp_dir,
        shard_dir=f"{temp_dir}/shards",
    )

    try:
        episodes = system['episodes']
        vagus = system['vagus']
        dream = system['dream']

        # Store test episodes
        for i in range(5):
            state = create_test_state(['neutral', 'crisis', 'emergence'][i % 3])
            episode = EnhancedEpisode(
                prompt=f"Dream test {i}",
                reply=f"Reply {i}",
                state=state,
                quality=0.6 + (i % 3) * 0.1,
            )
            await episodes.store_episode(episode)

        results.ok("dream_setup")

        # Test cycle
        vagus.set_simulated_state(create_test_state('emergence'))
        stats = await dream.force_cycle()

        if stats.mode == 'CONSOLIDATING':
            results.ok("dream_cycle_mode")
        else:
            results.fail("dream_cycle_mode", f"expected CONSOLIDATING, got {stats.mode}")

        # Check state
        state = dream.get_state()
        if state['total_cycles'] >= 1:
            results.ok("dream_state")
        else:
            results.fail("dream_state", "no cycles recorded")

    finally:
        await close_dream_system(system)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 LIMPHA FULL TEST SUITE 🧪")
    print("=" * 60)

    results = TestResults()

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Run tests
        await test_vagus_connector(results)

        await test_episodes_enhanced(
            results,
            os.path.join(temp_dir, 'episodes.db')
        )

        await test_consolidation(
            results,
            os.path.join(temp_dir, 'consolidation.db')
        )

        await test_graph_memory(
            results,
            os.path.join(temp_dir, 'graph.db')
        )

        await test_search(
            results,
            os.path.join(temp_dir, 'search.fts')
        )

        await test_shard_bridge(
            results,
            os.path.join(temp_dir, 'shards.db'),
            os.path.join(temp_dir, 'shards')
        )

        await test_dream_loop(results, temp_dir)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    # Summary
    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
