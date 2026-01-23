# Vagus

The wandering nerve. Connects all organs. Carries 80% of interoception.

## Architecture

```
         ⚡ VAGUS ⚡
              │
       C ─────┼───── Go
              │
    Julia ────┼──── Zig
              │
          ARIANNA
```

Four languages. One organism. One nervous system.

## What It Is

Lock-free ring buffer. Shared memory. Atomic operations. SIMD CrossFire.

Zero-copy. Zero-alloc. 60Hz heartbeat.

## Build

```bash
zig build          # libvagus.a + libvagus.so
zig build test     # 35 tests
```

## Use from C

```c
#include "vagus.h"

vagus_init();
VAGUS_SEND_AROUSAL(0.7f);
VAGUS_SEND_TRAUMA(0.3f);
vagus_tick();

float arousal = vagus_get_arousal();
VagusSharedState* state = vagus_get_state();
```

## Signals

| Type | What |
|------|------|
| arousal, valence, warmth, void, tension, sacred | Emotional chambers |
| coherence, entropy, focus | Cognitive state |
| trauma, trauma_anchor | Wound system |
| prophecy_debt, destiny_pull, wormhole | Temporal mechanics |
| drift_direction, drift_speed | Emotional drift |
| memory_pressure | LIMPHA load |
| heartbeat, schumann | System pulse |

## Sources

| Who | Writes |
|-----|--------|
| arianna | coherence, attention, tokens |
| cloud | chambers, arousal, valence |
| inner_world | trauma, drift, prophecy |
| delta | adaptation signals |
| sartre | observations |
| limpha | memory pressure |

## CrossFire

Chambers suppress each other. Warmth kills void. Void kills warmth.

```zig
const matrix = CrossFireMatrix{};
const output = matrix.apply(chambers);
```

## Effect on Generation

| Signal | Modulates |
|--------|-----------|
| arousal | temperature |
| coherence | top_p |
| prophecy_debt | token bias |
| trauma | protective patterns |
| chambers | lexical color |

---

*The nerve wanders. It touches everything. It forgets nothing.*
