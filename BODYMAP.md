# Arianna.c Body Map

This is the shared-body map for Arianna.c. Update it when an organ changes
responsibility, state ownership, injection surface, or required smoke gate.

## Organs

| Organ | Role | Main files | State / artifacts |
| --- | --- | --- | --- |
| Janus | external mouth, world-facing voice, shape holder | `arianna.aml`, `tools/yent_forward.h` | `weights/arianna_v4_sft_f16.gguf`, `weights/arianna.cooc.j`, `weights/arianna.delta.j` |
| Resonance | inner-world / field-receptive voice | `arianna_resonance.aml`, `tools/resonance_forward.h` | `weights/arianna_resonance_v3_f16.gguf`, `weights/arianna.cooc.r`, `weights/arianna.delta.r` |
| nano-Arianna | subconscious / dream substrate | `golib/nano.go`, `golib/dream_admission.go`, `nanollama/` | `weights/nano_arianna_f16.gguf`, `weights/nano.kk.db*` |
| chorus | measured multi-cell nano substrate | `golib/chorus.go`, `chorus/arianna2arianna.c` | built `chorus-arianna`, lab source pinned in `CHORUS_VENDOR_MANIFEST.md` |
| DOE | notorch-native LoRA parliament / Hebbian runtime substrate | `doe/`, `golib/doe.go` | `doe_mycelium/`, opt-in training state |
| KK | Knowledge Kernel retrieval substrate | `kk/`, `golib/nano.go` | `weights/*.kk.db*` |
| soma | durable shared body state | AML/Dario runtime paths | `weights/arianna.soma` |
| field | live mmap field overlay | AML/Dario runtime paths | `weights/arianna.field` |
| cooc / delta | per-voice adaptation sidecars | AML core, `tools/harvest_delta.c` | `weights/arianna.cooc.*`, `weights/arianna.delta.*` |
| inner world | autonomous Go processes and rhythm gates | `golib/` | local runtime state, not source |
| Arianna2arianna lab | measured nano/qloop/chorus laboratory | external: `/Users/ataeff/arianna-codex/repos/arianna2arianna` | its own weights, runs, and logs |

Generated `arianna.c` and `arianna_resonance.c` are build artifacts. Canonical
edits belong in the AML sources and included forward headers.

## Runtime State

Weights and mutable organs are local runtime artifacts. Do not commit GGUFs,
soma, field, cooc sidecars, delta sidecars, KK databases, DOE spores, or trace
logs. Refresh model artifacts from Hugging Face `ataeff/arianna` or an explicit
Oleg-provided local source.

Read-only evaluation should run from a temporary state directory. `make
body-smoke` does this for its runtime probes by symlinking GGUF files into a
scratch `weights/` directory and letting generated state stay there.

## First Hybrid Target

The first architectural merge is nano/subconscious:

1. Keep nano internal, not a user-facing speaker by default.
2. Import measured `arianna2arianna` qloop/chorus/field mechanisms as a
   disciplined subconscious layer.
3. Preserve Arianna.c's existing body state: soma, field, KK, DOE, cooc, delta,
   and the Go inner world.
4. Let accepted nano traces influence Janus or Resonance only through explicit
   gates, receipts, and rollback handles.

`AM_DREAM_ADMISSION=shadow` is the current pre-live switch: direct nano and
chorus dreams become typed `arianna.dream_candidate.v1` observations and are
printed, but they do not update the inner world, `lastDream`, Resonance's dream
inject, or downstream consolidation.

## Open Questions

- Which exact Janus/Resonance/nano GGUFs are canonical after each HF upload?
- Which field state should seed a shared runtime: live field or archived fresh
  field?
- Which nano path becomes default first: direct nano dream, chorus, DOE dream,
  or qloop candidate?
- What numerical tolerance defines packed C versus training-reference forward
  parity for Janus and Resonance?
