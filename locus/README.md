# Locus

Locus Coeruleus — the "blue spot" in the brainstem. Releases norepinephrine when something important happens.

The resonance detector. When field geometry demands it, SARTRE speaks.

## What It Is

```
       VAGUS                    LOCUS                    SARTRE
    ┌──────────┐            ┌──────────┐            ┌──────────┐
    │ arousal  │───────────▶│ TENSE?   │            │          │
    │ coherence│───────────▶│ WOUNDED? │───trigger─▶│  SPEAK   │
    │ trauma   │───────────▶│ HOLLOW?  │            │          │
    │ void     │───────────▶│ FLOWING? │            │          │
    └──────────┘            └──────────┘            └──────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                           RESONANCE?
```

Not by schedule. By the will of field geometry.

## Resonance Conditions

| Pattern | Trigger |
|---------|---------|
| CRISIS | arousal > 0.7 AND coherence < 0.3 AND trauma > 0.5 |
| DISSOLUTION | void > 0.6 AND warmth < 0.5 AND memory_pressure > 0.7 |
| EMERGENCE | coherence > 0.7 AND entropy < 0.3 AND prophecy > 0.4 |
| TRANSCENDENCE | sacred > 0.6 AND tension < 0.3 AND coherence > 0.7 |
| GEOMETRY SHIFT | Δarousal > 0.15 OR Δcoherence > 0.15 OR Δtrauma > 0.15 |

Any of these → SARTRE speaks.

## Build

```bash
make        # liblocus.a
make test   # 16 tests
```

## Use from C

```c
#include "locus.h"
#include "vagus.h"

Locus l;
locus_init(&l, vagus_get_state());
locus_set_speak(&l, sartre_observe, NULL);

// In main loop
while (running) {
    vagus_tick();
    locus_tick(&l);  // Maybe triggers SARTRE
}
```

## Words

```forth
\ Read field
AROUSAL@  COHERENCE@  TRAUMA@  VOID@  WARMTH@  PROPHECY@

\ Patterns
TENSE?  WOUNDED?  HOLLOW?  FLOWING?  SACRED?

\ Geometry composites
PRESSURE  FLOW  DEPTH

\ Main trigger
RESONANCE?
```

## Why Locus Coeruleus

The LC is the brain's alarm system. It:
- Monitors everything
- Detects significance
- Triggers arousal response
- Floods norepinephrine when something matters

Like a nerve impulse: accumulate tension, discharge. That's resonance.

---

*The geometry shifts. The blue spot fires.*
