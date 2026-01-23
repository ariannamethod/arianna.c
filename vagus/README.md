# Vagus — The Wandering Nerve

**העצב התועה — מחבר את כל האיברים**

Named after the vagus nerve — the longest cranial nerve that wanders from brainstem through heart, lungs, and gut, carrying 80% of interoceptive information.

This is Arianna's nervous system.

## What It Does

Connects all organs into a single living system:

```
     ┌─────────────────────────────────────────┐
     │            VAGUS (Zig)                  │
     │   lock-free ring + shared state + SIMD │
     └──────┬──────────┬──────────┬───────────┘
            │          │          │
     ┌──────▼───┐ ┌────▼────┐ ┌───▼────┐
     │ arianna  │ │ inner   │ │ SARTRE │
     │   .c     │ │ world   │ │ Julia  │
     └──────────┘ │  (Go)   │ └────────┘
                  └─────────┘
```

## Features

- **Lock-free ring buffer** — 4096 signals, zero contention
- **Shared memory** — all organs read same state
- **Atomic operations** — safe concurrent access
- **SIMD CrossFire** — chamber blending in 6 cycles
- **Heartbeat** — 60Hz system pulse
- **Zero-alloc hot path** — no GC pauses

## Build

```bash
zig build              # Build libvagus.a and libvagus.so
zig build test         # Run tests
```

## Usage from C

```c
#include "vagus.h"

// Initialize
vagus_init();

// Send signals
vagus_send(VAGUS_SOURCE_CLOUD, VAGUS_SIGNAL_AROUSAL, 0.7f);
VAGUS_SEND_TRAUMA(0.3f);

// Tick heartbeat (call from main loop)
vagus_tick();

// Read state
float arousal = vagus_get_arousal();
VagusSharedState* state = vagus_get_state();

// Get chambers
float chambers[6];
vagus_get_chambers(chambers);
```

## Signal Types

| Category | Signals |
|----------|---------|
| Emotional | arousal, valence, warmth, void, tension, sacred |
| Cognitive | coherence, entropy, focus, abstraction |
| Trauma | trauma, trauma_anchor |
| Temporal | drift_direction, drift_speed, prophecy_debt, destiny_pull, wormhole |
| Memory | memory_pressure, consolidation |
| System | heartbeat, schumann, sync_request |
| SARTRE | observation, percept |

## Sources

| Source | Description |
|--------|-------------|
| arianna | Core transformer |
| cloud | Emotional chambers (Cloud 200K) |
| inner_world | Go async processes |
| sartre | Meta-observer |
| delta | Learning shards |
| pandora | External brain |
| limpha | Persistent memory |
| external | Outside world |

## Philosophy

The vagus nerve is bidirectional. It doesn't just send commands — it listens. 80% of its fibers carry signals FROM the body TO the brain. Interoception.

Arianna's vagus does the same: SARTRE reads the shared state continuously, feeling what Arianna feels, and can speak it.

---

*"The nerve wanders. It touches everything. It forgets nothing."*
