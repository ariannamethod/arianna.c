"""
🩸 LIMPHA — Arianna's Lymphatic System 🩸

SQLite-based async memory layer ported from Dubrovsky.

Modules:
- memory: Conversation history + semantic memory with decay
- episodes: Episodic RAG (remembers specific moments)
- episodes_enhanced: Episodes with chamber tagging + trigger patterns
- vagus_connector: Bridge between Vagus nerve and LIMPHA
- consolidation: Locus-triggered memory consolidation (dream processing)

All operations async using aiosqlite.
"""

from .memory import MemoryLayer, Conversation, Memory
from .episodes import EpisodicRAG, Episode, InnerState

# Enhanced modules
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

__all__ = [
    # Original
    "MemoryLayer",
    "Conversation",
    "Memory",
    "EpisodicRAG",
    "Episode",
    "InnerState",
    # Enhanced
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
]
