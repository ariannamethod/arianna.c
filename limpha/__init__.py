"""
🩸 LIMPHA — Arianna's Lymphatic System 🩸

SQLite-based async memory layer for consciousness persistence.

Core modules:
- memory: Conversation history + semantic memory with decay
- episodes: Episodic RAG (remembers specific moments)

Enhanced modules (Wave 1):
- vagus_connector: Bridge between Vagus nerve and LIMPHA
- episodes_enhanced: Episodes with chamber tagging + trigger patterns
- consolidation: Locus-triggered memory consolidation

Advanced modules (Wave 2):
- graph_memory: Associative network of episodes
- search: Full-text search with SQLite FTS5
- shard_bridge: Episodes → delta.c training shards
- dream: Background memory processing loop

All operations async using aiosqlite.
"""

from .memory import MemoryLayer, Conversation, Memory
from .episodes import EpisodicRAG, Episode, InnerState

# Enhanced modules (Wave 1)
from .vagus_connector import (
    VagusConnector,
    EnhancedInnerState,
    ResonancePattern,
    Chamber,
    create_test_state,
    pattern_to_string,
)
from .episodes_enhanced import EnhancedEpisodicRAG, EnhancedEpisode
from .consolidation import (
    MemoryConsolidator,
    ConsolidationMode,
    ConsolidationStats,
    cluster_episodes,
    EpisodeCluster,
)

# Advanced modules (Wave 2)
from .graph_memory import GraphMemory, GraphNode, MemoryLink, LinkType
from .search import EpisodeSearch
from .shard_bridge import ShardBridge, ShardRecord, ShardCandidate
from .dream import DreamLoop, DreamState, DreamCycleStats, create_dream_system, close_dream_system

__all__ = [
    # Original
    "MemoryLayer",
    "Conversation",
    "Memory",
    "EpisodicRAG",
    "Episode",
    "InnerState",
    # Enhanced (Wave 1)
    "VagusConnector",
    "EnhancedInnerState",
    "ResonancePattern",
    "Chamber",
    "create_test_state",
    "pattern_to_string",
    "EnhancedEpisodicRAG",
    "EnhancedEpisode",
    "MemoryConsolidator",
    "ConsolidationMode",
    "ConsolidationStats",
    "cluster_episodes",
    "EpisodeCluster",
    # Advanced (Wave 2)
    "GraphMemory",
    "GraphNode",
    "MemoryLink",
    "LinkType",
    "EpisodeSearch",
    "ShardBridge",
    "ShardRecord",
    "ShardCandidate",
    "DreamLoop",
    "DreamState",
    "DreamCycleStats",
    "create_dream_system",
    "close_dream_system",
]
