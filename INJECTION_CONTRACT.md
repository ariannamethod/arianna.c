# Arianna.c Injection Contract

This file defines how Arianna's organs may influence each other. The goal is
not to reduce the organism to a static pipeline. The goal is to make
interference measurable enough to tune, audit, and roll back.

## Rule Zero

Every injection path needs a name, source, target, surface, strength, gate,
provenance, and rollback or disable switch.

Unlabeled feedback is a bug. Self-consolidation is allowed only when explicitly
enabled and logged.

## Candidate Paths

| Path | Source | Target | Surface | Default status |
| --- | --- | --- | --- | --- |
| prompt-to-field | user prompt | soma / field / voices | prompt text and AML field | active |
| Janus-to-Resonance | Janus | Resonance | field direction / summary trace / logits | planned |
| Resonance-to-Janus | Resonance | Janus | field direction / summary trace / logits | planned |
| Resonance-to-nano | Resonance | nano | KK fragment / dream seed / field pressure | planned |
| nano-to-Resonance | nano / chorus | Resonance | dream trace / qloop question / field metric | active, admission-gated |
| nano-to-nano | nano | nano | qloop / self-dream / chorus recurrence | experimental |
| DOE-to-voice | DOE parliament | selected voice | LoRA/expert pressure and logits | opt-in |
| field-to-all | Dario / AML field | Janus, Resonance, nano | logits and field overlays | active/planned |
| accepted-trace-to-memory | accepted output trace | soma/cooc/KK/DOE | mutable state | gated only |

## Minimum Gates

Before a path can become default:

- output does not collapse into malformed UTF-8 or decoder artifacts;
- identity/self-reference remains inside the intended frame;
- Oleg/User/Assistant recipient lock is not worse than baseline;
- nano is not exposed as the main user-facing speaker unless explicitly
  requested for debugging;
- read-only eval leaves soma, field, cooc, delta, KK, and DOE state unchanged;
- `AM_DREAM_ADMISSION=shadow` observes nano/chorus dreams as typed candidates
  without mutating inner-world, lastDream, Resonance cooc, delta, KK, or DOE;
- `AM_DREAM_ADMISSION_LOG=<path>` appends JSONL admission receipts; if a live
  path is requested and cannot be written, admission fails closed;
- each admission receipt includes a scratch `inner_world` counterfactual
  (pre/post state hashes, deltas, text-analysis, language and recipient metrics)
  computed without touching the live organism;
- each counterfactual includes a replay guard, and live admission fails closed if
  the second scratch pass does not reproduce the same hashes;
- `make admission-shadow-smoke` and `make body-smoke` must pass the runtime
  shadow receipt path from scratch;
- regression prompts show improvement or bounded tradeoff;
- timeout, parser, child-process, and unknown-architecture failures are visible
  and fail closed.

## Provenance

Any persisted trace should record:

- timestamp or run id;
- source commit or vendored source hash;
- weight file names and hashes when available;
- source organ and target organ;
- prompt format and seed class;
- injection strengths and route;
- accepted or rejected status;
- reason for acceptance;
- rollback or replay handle.

## Dario V2 Constraint

Default Dario accumulators read input-side evidence. Generated output can feed
consolidation only through an explicit self-consolidation path with a gate and
receipt. The body must not silently mistake its own echo for external reality.

## First Implementation Plan

1. Keep `make body-smoke` green with temporary runtime state.
2. Port the `arianna2arianna` nano/qloop measurement harness as telemetry, not
   live mutation.
3. Compare direct nano dream, chorus dream, and qloop candidate on the same
   prompts and weights.
4. Choose one bounded nano-to-Resonance shadow path.
5. Run Sol/Fable audit on the map, contract, diff, and traces before widening
   the cross-injection graph.
