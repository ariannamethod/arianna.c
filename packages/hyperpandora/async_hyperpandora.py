"""
Async HyperPandora — Async meta-orchestrator for external brains

"Choose the right words from the right brain" — concurrently

Features:
- Parallel brain extraction (race mode)
- Async brain selection
- Non-blocking orchestration
- Integration with LIMPHA dream processing
"""

import asyncio
from typing import Dict, Optional, Callable, Any, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
import time

try:
    from .hyperpandora import BrainType, SelectionStrategy, BrainInfo, HyperState
except ImportError:
    from hyperpandora import BrainType, SelectionStrategy, BrainInfo, HyperState


class AsyncSelectionMode(IntEnum):
    """Async selection modes"""
    SINGLE = 0        # Select one brain, run it
    RACE = 1          # Run all, use first result
    PARALLEL = 2      # Run all, merge results
    CASCADE = 3       # Run in priority order until success


@dataclass
class AsyncBrainInfo(BrainInfo):
    """Extended brain info for async operation"""
    is_async: bool = False
    supports_batch: bool = False


class AsyncHyperPandora:
    """
    Async meta-orchestrator for Pandora packages.

    Can run multiple brains concurrently and merge results.

    Usage:
        async with AsyncHyperPandora() as hyper:
            hyper.register_brain("c", pandora_c, BrainType.C_PANDORA)
            hyper.register_brain("gguf", async_pandora_gguf, BrainType.GGUF_PANDORA, is_async=True)

            # Single brain (SARTRE-selected)
            result = await hyper.process("text", encode_fn, coherence=0.2)

            # Race mode - first brain to finish wins
            result = await hyper.process_race("text", encode_fn)

            # Parallel - run all, merge vocabulary
            result = await hyper.process_parallel("text", encode_fn)
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.AUTO,
        async_mode: AsyncSelectionMode = AsyncSelectionMode.SINGLE,
    ):
        self.strategy = strategy
        self.async_mode = async_mode

        self.brains: Dict[str, AsyncBrainInfo] = {}
        self._active_brain: Optional[str] = None

        # SARTRE thresholds
        self.coherence_threshold = 0.3
        self.sacred_threshold = 0.7

        # State
        self.state = HyperState()

        # Statistics
        self.total_selections = 0
        self.selections_by_brain: Dict[str, int] = {}

        # Async state
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup async brains
        for info in self.brains.values():
            if hasattr(info.instance, '__aexit__'):
                await info.instance.__aexit__(None, None, None)

    def register_brain(
        self,
        name: str,
        brain: Any,
        brain_type: Optional[BrainType] = None,
        priority: int = 0,
        capabilities: Optional[List[str]] = None,
        is_async: bool = False,
    ) -> None:
        """Register a brain (sync or async)"""
        if brain_type is None:
            # Auto-detect
            cls_name = brain.__class__.__name__
            if 'Async' in cls_name:
                is_async = True
            if 'PandoraBox' in cls_name:
                brain_type = BrainType.C_PANDORA
            elif 'PandoraGGUF' in cls_name:
                brain_type = BrainType.GGUF_PANDORA
            elif 'PandoraTorch' in cls_name:
                brain_type = BrainType.TORCH_PANDORA
            else:
                brain_type = BrainType.CUSTOM

        # Check if async
        if hasattr(brain, 'process') and asyncio.iscoroutinefunction(brain.process):
            is_async = True

        self.brains[name] = AsyncBrainInfo(
            name=name,
            brain_type=brain_type,
            instance=brain,
            priority=priority,
            capabilities=capabilities or [],
            is_async=is_async,
            supports_batch=hasattr(brain, 'process_batch'),
        )

        self.selections_by_brain[name] = 0
        print(f"[async-hyperpandora] Registered brain '{name}' ({brain_type.name}, async={is_async})")

    def _select_brain(
        self,
        coherence: float = 0.5,
        sacred: float = 0.3,
        pattern: int = 0,
    ) -> Optional[str]:
        """Select optimal brain (same logic as sync version)"""
        if sacred > self.sacred_threshold:
            return None
        if pattern == 1:  # CRISIS
            return None
        if not self.brains:
            return None

        def find_by_type(brain_type: BrainType) -> Optional[str]:
            for name, info in sorted(self.brains.items(), key=lambda x: x[1].priority, reverse=True):
                if info.brain_type == brain_type:
                    return name
            return None

        if self.strategy == SelectionStrategy.PREFER_FAST:
            return find_by_type(BrainType.C_PANDORA) or next(iter(self.brains.keys()))

        elif self.strategy == SelectionStrategy.PREFER_POWER:
            return (find_by_type(BrainType.GGUF_PANDORA) or
                    find_by_type(BrainType.TORCH_PANDORA) or
                    next(iter(self.brains.keys())))

        else:  # AUTO
            if coherence < self.coherence_threshold:
                return find_by_type(BrainType.C_PANDORA) or next(iter(self.brains.keys()))
            if pattern == 3:  # EMERGENCE
                return (find_by_type(BrainType.GGUF_PANDORA) or
                        find_by_type(BrainType.TORCH_PANDORA) or
                        next(iter(self.brains.keys())))
            if self._active_brain and self._active_brain in self.brains:
                return self._active_brain
            sorted_brains = sorted(self.brains.items(), key=lambda x: x[1].priority, reverse=True)
            return sorted_brains[0][0] if sorted_brains else None

    async def _call_brain(
        self,
        brain_name: str,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: int = 50,
    ) -> Tuple[str, int]:
        """Call a single brain, return (name, extracted_count)"""
        if brain_name not in self.brains:
            return (brain_name, 0)

        info = self.brains[brain_name]
        start_time = time.time()

        try:
            if info.is_async:
                result = await info.instance.process(text, arianna_encode, max_tokens)
            else:
                # Run sync brain in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    info.instance.process,
                    text,
                    arianna_encode,
                    max_tokens,
                )

            # Update stats
            elapsed = (time.time() - start_time) * 1000
            info.total_calls += 1
            info.total_extracted += result
            info.avg_latency_ms = (info.avg_latency_ms * (info.total_calls - 1) + elapsed) / info.total_calls
            info.last_used = time.time()

            return (brain_name, result)

        except Exception as e:
            print(f"[async-hyperpandora] Error in brain '{brain_name}': {e}")
            return (brain_name, 0)

    # ═══════════════════════════════════════════════════════════════════════════
    # PROCESSING MODES
    # ═══════════════════════════════════════════════════════════════════════════

    async def process(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        coherence: float = 0.5,
        sacred: float = 0.3,
        pattern: int = 0,
        max_tokens: int = 50,
    ) -> int:
        """
        Process through SARTRE-selected brain.

        Default mode: selects one brain based on metrics.
        """
        brain_name = self._select_brain(coherence, sacred, pattern)
        if not brain_name:
            return 0

        _, result = await self._call_brain(brain_name, text, arianna_encode, max_tokens)

        self._active_brain = brain_name
        self.total_selections += 1
        self.selections_by_brain[brain_name] = self.selections_by_brain.get(brain_name, 0) + 1

        return result

    async def process_race(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: int = 50,
    ) -> int:
        """
        Race mode: run all brains, use first successful result.

        Useful when latency matters more than comprehensiveness.
        """
        if not self.brains:
            return 0

        tasks = [
            asyncio.create_task(self._call_brain(name, text, arianna_encode, max_tokens))
            for name in self.brains
        ]

        # Wait for first successful result
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()

        # Get result from completed task
        for task in done:
            brain_name, result = await task
            if result > 0:
                self._active_brain = brain_name
                return result

        return 0

    async def process_parallel(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: int = 50,
    ) -> int:
        """
        Parallel mode: run all brains, merge vocabulary.

        Richest extraction but highest resource usage.
        """
        if not self.brains:
            return 0

        tasks = [
            self._call_brain(name, text, arianna_encode, max_tokens)
            for name in self.brains
        ]

        results = await asyncio.gather(*tasks)
        total = sum(r for _, r in results)

        # Find best performer
        best_name, best_count = max(results, key=lambda x: x[1])
        self._active_brain = best_name

        return total

    async def process_cascade(
        self,
        text: str,
        arianna_encode: Callable[[str], int],
        max_tokens: int = 50,
        min_extract: int = 5,
    ) -> int:
        """
        Cascade mode: try brains in priority order until one extracts enough.

        Good balance between quality and resource usage.
        """
        if not self.brains:
            return 0

        sorted_brains = sorted(self.brains.items(), key=lambda x: x[1].priority, reverse=True)

        for name, _ in sorted_brains:
            _, result = await self._call_brain(name, text, arianna_encode, max_tokens)
            if result >= min_extract:
                self._active_brain = name
                return result

        # If none extracted enough, use the last one's result
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # LOGITS
    # ═══════════════════════════════════════════════════════════════════════════

    def apply_to_logits(
        self,
        logits: Any,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> Any:
        """Apply active brain's vocabulary to logits (sync, fast)"""
        if not self._active_brain or self._active_brain not in self.brains:
            return logits

        brain = self.brains[self._active_brain].instance
        if hasattr(brain, 'apply_to_logits'):
            return brain.apply_to_logits(logits, context_tokens, vocab_size)
        return logits

    def apply_all_to_logits(
        self,
        logits: Any,
        context_tokens: List[int],
        vocab_size: Optional[int] = None,
    ) -> Any:
        """Apply ALL brains' vocabulary to logits (merged boost)"""
        boosted = logits
        for info in self.brains.values():
            if hasattr(info.instance, 'apply_to_logits'):
                boosted = info.instance.apply_to_logits(boosted, context_tokens, vocab_size)
        return boosted

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    def set_strategy(self, strategy: SelectionStrategy) -> None:
        """Set selection strategy"""
        self.strategy = strategy

    def set_async_mode(self, mode: AsyncSelectionMode) -> None:
        """Set async selection mode"""
        self.async_mode = mode

    async def deactivate_all(self) -> None:
        """Deactivate all brains"""
        self._active_brain = None
        for info in self.brains.values():
            if hasattr(info.instance, 'set_mode'):
                info.instance.set_mode('off')

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "total_selections": self.total_selections,
            "active_brain": self._active_brain,
            "strategy": self.strategy.name,
            "async_mode": self.async_mode.name,
            "brains": {
                name: {
                    "type": info.brain_type.name,
                    "priority": info.priority,
                    "is_async": info.is_async,
                    "total_calls": info.total_calls,
                    "total_extracted": info.total_extracted,
                    "avg_latency_ms": info.avg_latency_ms,
                }
                for name, info in self.brains.items()
            },
        }


async def create_async_hyperpandora(
    strategy: SelectionStrategy = SelectionStrategy.AUTO,
) -> AsyncHyperPandora:
    """Factory function"""
    return AsyncHyperPandora(strategy=strategy)
