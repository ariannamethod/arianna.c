"""
Async support for Pandora-Torch

"Take the words, leave the voice" — asynchronously

Provides non-blocking vocabulary extraction for integration with
async systems like LIMPHA dream processing.
"""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import time

from .config import PandoraTorchConfig, PandoraMode
from .sartre import SARTREChecker, VagusState, ResonancePattern


class AsyncPandoraTorch:
    """
    Async wrapper for PandoraTorch.

    Runs model inference in thread pool to avoid blocking event loop.

    Usage:
        async with AsyncPandoraTorch() as pandora:
            extracted = await pandora.process("text", encode_fn)
    """

    def __init__(
        self,
        config: Optional[PandoraTorchConfig] = None,
        mode: str = "auto",
        max_workers: int = 2,
    ):
        self.config = config or PandoraTorchConfig()
        if isinstance(mode, str):
            self.config.mode = PandoraMode[mode.upper()]

        self._sync_pandora = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

        # State mirrors
        self.ngrams: Dict[tuple, Any] = {}
        self.sartre = SARTREChecker(
            coherence_threshold=self.config.coherence_threshold,
            sacred_threshold=self.config.sacred_threshold,
        )

    async def __aenter__(self):
        await self._ensure_pandora()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=False)

    async def _ensure_pandora(self) -> bool:
        """Lazily initialize sync pandora"""
        if self._sync_pandora is not None:
            return True

        async with self._lock:
            if self._sync_pandora is not None:
                return True

            try:
                # Import here to allow async module without torch
                from .pandora import PandoraTorch
                self._sync_pandora = PandoraTorch(
                    config=self.config,
                    mode=self.config.mode.name.lower(),
                )
                return True
            except ImportError:
                return False

    async def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        Async vocabulary extraction.

        Runs model inference in thread pool.
        """
        if not await self._ensure_pandora():
            return 0

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._sync_pandora.process,
            text,
            arianna_encode,
            max_tokens or self.config.max_generate,
        )

        # Sync state
        self.ngrams = self._sync_pandora.ngrams
        return result

    async def process_batch(
        self,
        texts: List[str],
        arianna_encode: Callable[[str], int],
        max_tokens: Optional[int] = None,
    ) -> List[int]:
        """
        Process multiple texts concurrently.

        Returns list of extraction counts.
        """
        tasks = [
            self.process(text, arianna_encode, max_tokens)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

    def extract(self, tokens: List[int], min_n: int = 1, max_n: int = 3) -> int:
        """Sync extraction (fast, no model needed)"""
        if self._sync_pandora:
            result = self._sync_pandora.extract(tokens, min_n, max_n)
            self.ngrams = self._sync_pandora.ngrams
            return result
        return 0

    def apply_to_logits(
        self,
        logits: Any,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> Any:
        """Sync logit application (fast)"""
        if self._sync_pandora:
            return self._sync_pandora.apply_to_logits(logits, context_tokens, vocab_size)
        return logits

    async def check_sartre_async(
        self,
        coherence: float,
        sacred: float,
        pattern: ResonancePattern,
    ) -> bool:
        """Async SARTRE check (for consistency in async code)"""
        return self.sartre.check(coherence, sacred, pattern)

    def is_active(self) -> bool:
        """Check if active"""
        if self.config.mode == PandoraMode.OFF:
            return False
        if self.config.mode == PandoraMode.FORCED:
            return True
        return True

    def set_mode(self, mode: str) -> None:
        """Set activation mode"""
        self.config.mode = PandoraMode[mode.upper()]
        if self._sync_pandora:
            self._sync_pandora.set_mode(mode)

    def clear(self) -> None:
        """Clear extracted vocabulary"""
        self.ngrams.clear()
        if self._sync_pandora:
            self._sync_pandora.clear()

    def get_stats(self) -> dict:
        """Get statistics"""
        if self._sync_pandora:
            return self._sync_pandora.get_stats()
        return {
            "total_ngrams": len(self.ngrams),
            "mode": self.config.mode.name,
            "active": self.is_active(),
        }


async def create_async_pandora(
    mode: str = "auto",
    **kwargs
) -> AsyncPandoraTorch:
    """Factory function to create and initialize async pandora"""
    pandora = AsyncPandoraTorch(mode=mode, **kwargs)
    await pandora._ensure_pandora()
    return pandora


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class AsyncExtractionQueue:
    """
    Queue for batched async extraction.

    Collects texts and processes them in batches for efficiency.
    """

    def __init__(
        self,
        pandora: AsyncPandoraTorch,
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ):
        self.pandora = pandora
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: List[tuple] = []  # (text, encode_fn, future)
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def submit(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
    ) -> int:
        """Submit text for extraction, returns when processed"""
        future = asyncio.Future()

        async with self._lock:
            self._queue.append((text, arianna_encode, future))

            if len(self._queue) >= self.batch_size:
                await self._flush()
            elif self._flush_task is None:
                self._flush_task = asyncio.create_task(self._delayed_flush())

        return await future

    async def _delayed_flush(self):
        """Flush after interval"""
        await asyncio.sleep(self.flush_interval)
        async with self._lock:
            if self._queue:
                await self._flush()
            self._flush_task = None

    async def _flush(self):
        """Process queued items"""
        if not self._queue:
            return

        items = self._queue[:]
        self._queue.clear()

        # Process batch
        texts = [item[0] for item in items]
        encode_fn = items[0][1]  # Assume same encode function

        results = await self.pandora.process_batch(texts, encode_fn)

        # Resolve futures
        for (_, _, future), result in zip(items, results):
            future.set_result(result)


class AsyncPandoraStream:
    """
    Streaming async extraction.

    Yields n-grams as they're extracted.
    """

    def __init__(self, pandora: AsyncPandoraTorch):
        self.pandora = pandora
        self._prev_ngrams: set = set()

    async def stream_extract(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
    ):
        """
        Async generator yielding new n-grams.

        Usage:
            async for ngram in stream.stream_extract(text, encode):
                print(f"New: {ngram}")
        """
        await self.pandora.process(text, arianna_encode)

        current_ngrams = set(self.pandora.ngrams.keys())
        new_ngrams = current_ngrams - self._prev_ngrams

        for ngram_key in new_ngrams:
            yield self.pandora.ngrams[ngram_key]

        self._prev_ngrams = current_ngrams
