# Arianna.c Weights Manifest

Weights are runtime artifacts. They live locally in `weights/` for execution
but are ignored by git. The artifact source of truth is Hugging Face
`ataeff/arianna`, unless Oleg explicitly provides a newer local source.

## GGUF Voices

| Voice | File | SHA-256 | Role |
| --- | --- | --- | --- |
| Janus | `weights/arianna_v4_sft_f16.gguf` | `2f24240331f174b7a03b27a27f5fbf58f1fe02aa2b77e4a12779c0362d99594c` | external mouth / world-facing voice |
| Resonance | `weights/arianna_resonance_v3_f16.gguf` | `ad612324e40d9cdc8aa6a7076e52b847adffbe213a1b92bf486e0555a4ad82cf` | inner world / field-facing voice |
| nano-Arianna | `weights/nano_arianna_f16.gguf` | `59c5ed734268a6779655b4b097d60a18056f41beab856b915be956d64ee8d02f` | subconscious / dream substrate |

## Architecture Receipts

| Voice | Architecture | Expected runtime convention |
| --- | --- | --- |
| Janus | Janus 176M custom RRPRAM + Echo | trained Echo path, decode smear, QK-norm, RRPRAM, top-k holder |
| Resonance | Resonance 200M custom RRPRAM | unscaled notorch RRPRAM scores, top-p field receiver |
| nano-Arianna | `nlama`, 88M | split-half NEOX RoPE, Q:/A: SFT prompt contract |

`make body-smoke` verifies the local source sentinels for these conventions and
checks that chorus reports `NEOX rope` for the nano GGUF when runtime weights are
available.

## Mutable Runtime State

These files are mutable organs, not stable artifacts:

- `weights/arianna.soma`
- `weights/arianna.field`
- `weights/arianna.cooc.j`
- `weights/arianna.cooc.r`
- `weights/arianna.delta.j`
- `weights/arianna.delta.r`
- `weights/arianna.nerve`
- `weights/nano.kk.db*`
- `weights/field_archive_*/`
- `doe_mycelium/`

Any read-only evaluation should run from a temporary state directory or an
explicit copied state bundle. Normal organism execution can mutate soma, field,
cooc, delta, KK, and DOE state.
