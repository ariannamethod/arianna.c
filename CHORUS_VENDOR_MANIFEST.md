# Chorus Vendor Manifest

Date: 2026-07-18

This file records the current vendoring boundary between the shared Arianna.c
body and the standalone arianna2arianna laboratory.

## Pinned Lab Snapshot

- Lab repo: `/Users/ataeff/arianna-codex/repos/arianna2arianna`
- Remote: `https://github.com/ariannamethod/arianna2arianna.git`
- Pinned commit: `dae3b9367eb75ce5b45bd95af29809b15fcfa98e`
- Subject: `the method hardens chorus gguf boundaries`
- Shared vendored file: `chorus/arianna2arianna.c`

## Current Integration State

The shared Arianna.c vendored chorus is byte-identical to the pinned lab
snapshot.

This snapshot brings the newer qloop/repl/direct-user machinery from the
standalone laboratory after replaying the shared C-hardening into that lab:

- `general.architecture=nlama` selects split-half NEOX RoPE;
- unknown architectures fail closed instead of guessing a RoPE convention;
- oversized/truncated GGUF strings, metadata reads, tensor dimensions, file
  offsets, tokenizer/embedding mismatch, and KV cache sizes are guarded;
- direct, field, and REPL paths are preserved from the lab snapshot.

## Validation Receipt

Standalone lab commit `dae3b93` was validated before vendoring:

- `make`
- `make test` -> `150 passed, 0 failed, 1 skipped`
- direct, field, and REPL smokes on `weights/nano_arianna_f16.gguf`

Do not overwrite this vendored file with an unverified lab snapshot. New lab
features should land here only after the same build, test, and direct/field/REPL
smoke sequence succeeds in the standalone repository.
