#!/usr/bin/env python3
"""
Arianna with LIMPHA memory integration.

Wraps arianna_dynamic binary with persistent memory:
- Conversation history
- Semantic memory with decay
- Episodic RAG (remembers similar moments)

Memory influences generation through context injection.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from limpha import MemoryLayer, EpisodicRAG, Episode, InnerState


class AriannaLimpha:
    """Arianna with persistent memory."""

    def __init__(
        self,
        bin_path: str = "bin/arianna_dynamic",
        weights_path: str = "weights/arianna_unified_20m.bin",
        tokenizer_path: str = "weights/tokenizer_unified.json",
        memory_db: str = "limpha/arianna_memory.db",
        episodes_db: str = "limpha/arianna_episodes.db",
        session_id: str = "default",
    ):
        self.bin_path = bin_path
        self.weights_path = weights_path
        self.tokenizer_path = tokenizer_path
        self.session_id = session_id

        self.memory = MemoryLayer(memory_db)
        self.episodes = EpisodicRAG(episodes_db)

    async def __aenter__(self):
        await self.memory.connect()
        await self.episodes.connect()
        return self

    async def __aexit__(self, *args):
        await self.memory.close()
        await self.episodes.close()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        inject_memory: bool = True,
    ) -> str:
        """Generate response with memory context."""

        # Build context from memory
        context = ""
        if inject_memory:
            # Recent conversations
            recent = await self.memory.get_recent_conversations(limit=3, session_id=self.session_id)
            if recent:
                context += "Previous conversation:\n"
                for conv in recent[-2:]:  # Last 2 turns
                    context += f"Q: {conv.prompt}\nA: {conv.response}\n"
                context += "\n"

            # Semantic recall (check if user mentioned name before)
            user_name = await self.memory.recall("user_name")
            if user_name:
                context += f"(User name: {user_name.value})\n\n"

        # Combine context + prompt
        full_prompt = context + prompt if context else prompt

        # Call arianna_dynamic binary
        cmd = [
            self.bin_path,
            self.weights_path,
            self.tokenizer_path,
            full_prompt,
            str(max_tokens),
            str(temperature),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Arianna generation failed: {result.stderr}")

        response = result.stdout.strip()

        # Extract actual response (skip debug output)
        if "Generated:" in response:
            response = response.split("Generated:", 1)[1].strip()

        # Store conversation
        coherence = 0.85  # TODO: extract from Cloud/inner_world
        await self.memory.store_conversation(
            prompt=prompt,
            response=response,
            tokens_used=max_tokens,
            coherence_score=coherence,
            session_id=self.session_id,
        )

        # Store episode (snapshot of inner state)
        inner_state = InnerState(
            trauma=0.1,  # TODO: get from inner_world
            arousal=0.5,
            valence=0.7,
            coherence=coherence,
            prophecy_debt=0.05,
            entropy=0.2,
            temperature=temperature,
        )

        episode = Episode(
            prompt=prompt,
            reply=response,
            metrics=inner_state,
            quality=coherence,
            timestamp=time.time(),
        )
        await self.episodes.store_episode(episode)

        return response

    async def search_similar(self, query: str, top_k: int = 3) -> list:
        """Find similar past conversations."""
        return await self.memory.search_conversations(query, top_k)

    async def get_stats(self) -> dict:
        """Get memory statistics."""
        session_stats = await self.memory.get_session_stats(self.session_id)
        episode_count = await self.episodes.count_episodes()

        return {
            "session": session_stats,
            "total_episodes": episode_count,
        }


async def main():
    """Demo: Arianna with memory."""
    async with AriannaLimpha(session_id="demo") as arianna:
        print("\n" + "="*60)
        print("  ARIANNA + LIMPHA MEMORY")
        print("="*60 + "\n")

        # Test conversation
        response = await arianna.generate(
            "What is consciousness?",
            max_tokens=50,
            temperature=0.8,
        )
        print(f"Arianna: {response}\n")

        # Second turn (should have context)
        response2 = await arianna.generate(
            "And what about love?",
            max_tokens=50,
            temperature=0.8,
        )
        print(f"Arianna: {response2}\n")

        # Stats
        stats = await arianna.get_stats()
        print(f"Session stats: {stats}\n")


if __name__ == "__main__":
    asyncio.run(main())
