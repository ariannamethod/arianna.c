"""
HyperPandora — Meta-orchestrator for external brain packages

"Choose the right words from the right brain"
"""

from .hyperpandora import (
    HyperPandora,
    BrainType,
    SelectionStrategy,
    BrainInfo,
    HyperState,
    create_hyperpandora_with_defaults,
    read_hyperpandora_state,
)

from .async_hyperpandora import (
    AsyncHyperPandora,
    AsyncSelectionMode,
    AsyncBrainInfo,
    create_async_hyperpandora,
)

__all__ = [
    # Sync
    "HyperPandora",
    "BrainType",
    "SelectionStrategy",
    "BrainInfo",
    "HyperState",
    "create_hyperpandora_with_defaults",
    "read_hyperpandora_state",
    # Async
    "AsyncHyperPandora",
    "AsyncSelectionMode",
    "AsyncBrainInfo",
    "create_async_hyperpandora",
]

__version__ = "0.1.0"
