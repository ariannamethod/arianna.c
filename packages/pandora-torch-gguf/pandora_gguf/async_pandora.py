"""
Async support for Pandora-Torch-GGUF

"Take the words from the tiny llama, leave the voice" — asynchronously

GGUF inference is CPU-bound, so we use thread pool for non-blocking operation.
"""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import PandoraGGUFConfig, PandoraMode
from .download import ensure_model


class AsyncPandoraGGUF:
    """
    Async wrapper for PandoraGGUF.

    Runs GGUF inference in thread pool since llama.cpp is CPU-bound.

    Usage:
        async with AsyncPandoraGGUF() as pandora:
            extracted = await pandora.process("text", encode_fn)
    """

    def __init__(
        self,
        config: Optional[PandoraGGUFConfig] = None,
        model_path: Optional[str] = None,
        auto_download: bool = True,
        mode: str = "auto",
        max_workers: int = 1,  # GGUF is memory-heavy, limit workers
    ):
        self.config = config or PandoraGGUFConfig()
        if model_path:
            self.config.model_path = model_path
        self.config.auto_download = auto_download

        if isinstance(mode, str):
            self.config.mode = PandoraMode[mode.upper()]

        self._sync_pandora = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        self._loading = False

        # State mirrors
        self.ngrams: Dict[tuple, Any] = {}

    async def __aenter__(self):
        await self._ensure_pandora()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=False)

    async def _ensure_pandora(self) -> bool:
        """Lazily initialize sync pandora with model loading"""
        if self._sync_pandora is not None:
            return True

        async with self._lock:
            if self._sync_pandora is not None:
                return True

            if self._loading:
                # Wait for another coroutine to finish loading
                while self._loading:
                    await asyncio.sleep(0.1)
                return self._sync_pandora is not None

            self._loading = True

            try:
                # Run model loading in thread pool (can take time)
                loop = asyncio.get_event_loop()
                self._sync_pandora = await loop.run_in_executor(
                    self._executor,
                    self._create_sync_pandora,
                )
                return self._sync_pandora is not None
            finally:
                self._loading = False

    def _create_sync_pandora(self):
        """Create sync pandora (runs in thread pool)"""
        try:
            from .pandora import PandoraGGUF
            pandora = PandoraGGUF(
                config=self.config,
                auto_download=self.config.auto_download,
                mode=self.config.mode.name.lower(),
            )
            # Trigger model load
            pandora._ensure_model()
            return pandora
        except Exception as e:
            print(f"[async-pandora-gguf] Failed to create: {e}")
            return None

    async def download_model(self) -> Path:
        """Async model download"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            ensure_model,
            Path(self.config.model_path),
            self.config.auto_download,
        )

    async def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        Async vocabulary extraction from TinyLlama.

        Runs GGUF inference in thread pool.
        """
        if not await self._ensure_pandora():
            return 0

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._sync_pandora.process,
            text,
            arianna_encode,
            max_tokens or self.config.max_tokens,
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
        Process multiple texts.

        Note: GGUF is memory-heavy, so we process sequentially by default.
        """
        results = []
        for text in texts:
            result = await self.process(text, arianna_encode, max_tokens)
            results.append(result)
        return results

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
            "model_loaded": self._sync_pandora is not None,
        }


async def create_async_pandora_gguf(
    mode: str = "auto",
    auto_download: bool = True,
    **kwargs
) -> AsyncPandoraGGUF:
    """Factory function to create and initialize async GGUF pandora"""
    pandora = AsyncPandoraGGUF(mode=mode, auto_download=auto_download, **kwargs)
    await pandora._ensure_pandora()
    return pandora
