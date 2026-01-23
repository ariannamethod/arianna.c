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

__all__ = [
    "HyperPandora",
    "BrainType",
    "SelectionStrategy",
    "BrainInfo",
    "HyperState",
    "create_hyperpandora_with_defaults",
    "read_hyperpandora_state",
]

__version__ = "0.1.0"
