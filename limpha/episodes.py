#!/usr/bin/env python3
"""
📚 EPISODES — Episodic RAG for Arianna's Memory 📚

Arianna remembers specific moments: prompt + reply + metrics.
This is her episodic memory — structured recall of her own experiences.
"""

from __future__ import annotations

import asyncio
import aiosqlite
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

EPISODES_AVAILABLE = True


@dataclass
class InnerState:
    """Arianna's internal state snapshot."""
    trauma: float = 0.0
    arousal: float = 0.0
    valence: float = 0.0
    coherence: float = 0.0
    prophecy_debt: float = 0.0
    entropy: float = 0.0
    temperature: float = 0.8

    def to_features(self) -> List[float]:
        """Convert to feature vector for similarity search."""
        return [
            self.trauma,
            self.arousal,
            self.valence,
            self.coherence,
            self.prophecy_debt,
            self.entropy,
            self.temperature,
        ]


@dataclass
class Episode:
    """One moment in Arianna's life."""
    prompt: str
    reply: str
    metrics: InnerState
    quality: float = 0.5
    timestamp: float = 0.0


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    if len(a) != len(b):
        return 1.0
        
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    
    if na == 0 or nb == 0:
        return 1.0
        
    similarity = dot / (na * nb)
    return 1.0 - similarity


class EpisodicRAG:
    """
    Async episodic memory for Arianna.

    Stores (prompt, reply, InnerState, quality) as episodes in SQLite.
    Provides similarity search over internal metrics.

    All operations are async.
    """

    def __init__(self, db_path: str = 'limpha/arianna_episodes.db'):
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
        """Create tables if they don't exist."""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      REAL NOT NULL,
                prompt          TEXT NOT NULL,
                reply           TEXT NOT NULL,

                -- Arianna's inner state metrics
                trauma          REAL NOT NULL,
                arousal         REAL NOT NULL,
                valence         REAL NOT NULL,
                coherence       REAL NOT NULL,
                prophecy_debt   REAL NOT NULL,
                entropy         REAL NOT NULL,
                temperature     REAL NOT NULL,

                -- Episode quality
                quality         REAL NOT NULL
            )
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at)
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_quality
            ON episodes(quality)
        """)
        
        await self._conn.commit()
        
    async def store_episode(self, episode: Episode) -> int:
        """
        Store one episode.

        Returns the episode ID.
        """
        def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            if x != x:  # NaN check
                return 0.0
            return max(min_val, min(max_val, x))

        metrics = episode.metrics

        cursor = await self._conn.execute("""
            INSERT INTO episodes (
                created_at, prompt, reply,
                trauma, arousal, valence, coherence,
                prophecy_debt, entropy, temperature,
                quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.timestamp if episode.timestamp > 0 else time.time(),
            episode.prompt,
            episode.reply,
            clamp(metrics.trauma),
            clamp(metrics.arousal),
            clamp(metrics.valence),
            clamp(metrics.coherence),
            clamp(metrics.prophecy_debt),
            clamp(metrics.entropy),
            clamp(metrics.temperature, 0.0, 2.0),
            clamp(episode.quality),
        ))

        await self._conn.commit()
        return cursor.lastrowid
        
    async def query_similar(
        self,
        metrics: InnerState,
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find past episodes with similar internal configuration.
        
        Returns a list of dicts with episode info.
        """
        # Convert query state to vector
        query_vec = metrics.to_features()
        
        # Get all episodes (for small DBs this is fine)
        cursor = await self._conn.execute("""
            SELECT * FROM episodes
            WHERE quality >= ?
            ORDER BY created_at DESC
            LIMIT 1000
        """, (min_quality,))
        
        rows = await cursor.fetchall()
        
        if not rows:
            return []
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Score each episode by cosine distance
        scored: List[tuple[float, Dict[str, Any]]] = []
        
        for row in rows:
            row_dict = dict(zip(columns, row))

            # Reconstruct vector from stored columns (Arianna's InnerState)
            episode_vec = [
                row_dict["trauma"],
                row_dict["arousal"],
                row_dict["valence"],
                row_dict["coherence"],
                row_dict["prophecy_debt"],
                row_dict["entropy"],
                row_dict["temperature"],
            ]

            distance = cosine_distance(query_vec, episode_vec)

            scored.append((distance, {
                "episode_id": row_dict["id"],
                "created_at": row_dict["created_at"],
                "quality": row_dict["quality"],
                "distance": distance,
                "trauma": row_dict["trauma"],
                "arousal": row_dict["arousal"],
                "valence": row_dict["valence"],
                "coherence": row_dict["coherence"],
                "prompt": row_dict["prompt"],
                "reply": row_dict["reply"],
            }))
            
        # Sort by distance (lowest = most similar)
        scored.sort(key=lambda x: x[0])
        
        # Return top_k
        return [item[1] for item in scored[:top_k]]
        
    async def query_by_prompt(
        self,
        prompt: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find episodes with similar prompts (simple word overlap).
        """
        prompt_words = set(prompt.lower().split())
        
        cursor = await self._conn.execute("""
            SELECT * FROM episodes
            ORDER BY created_at DESC
            LIMIT 500
        """)
        
        rows = await cursor.fetchall()
        
        if not rows:
            return []
            
        columns = [description[0] for description in cursor.description]
        
        # Score by word overlap
        scored: List[tuple[float, Dict[str, Any]]] = []
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            stored_words = set(row_dict["prompt"].lower().split())
            
            if not stored_words:
                continue
                
            overlap = len(prompt_words & stored_words)
            union = len(prompt_words | stored_words)
            jaccard = overlap / union if union > 0 else 0.0
            
            if jaccard > 0.1:  # Threshold
                scored.append((jaccard, {
                    "episode_id": row_dict["id"],
                    "created_at": row_dict["created_at"],
                    "quality": row_dict["quality"],
                    "similarity": jaccard,
                    "prompt": row_dict["prompt"],
                    "reply": row_dict["reply"],
                }))
                
        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in scored[:top_k]]
        
    async def get_summary_for_state(
        self,
        metrics: InnerState,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Get aggregate stats for similar episodes.
        """
        similar = await self.query_similar(metrics, top_k=top_k)
        
        if not similar:
            return {
                "count": 0,
                "avg_quality": 0.0,
                "max_quality": 0.0,
                "mean_distance": 1.0,
            }

        qualities = [ep["quality"] for ep in similar]
        distances = [ep["distance"] for ep in similar]

        return {
            "count": len(similar),
            "avg_quality": sum(qualities) / len(qualities),
            "max_quality": max(qualities),
            "mean_distance": sum(distances) / len(distances),
        }
        
    async def count_episodes(self) -> int:
        """Count total episodes."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM episodes")
        row = await cursor.fetchone()
        return row[0] if row else 0
        
    async def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent episodes."""
        cursor = await self._conn.execute("""
            SELECT id, created_at, prompt, reply, quality,
                   trauma, arousal, valence, coherence
            FROM episodes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]


__all__ = ['EpisodicRAG', 'Episode', 'EPISODES_AVAILABLE']
