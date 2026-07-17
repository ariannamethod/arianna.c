# Chorus Vendor Manifest

Date: 2026-07-18

This file records the current vendoring boundary between the shared Arianna.c
body and the standalone arianna2arianna laboratory.

## Pinned Lab Snapshot

- Lab repo: `/Users/ataeff/arianna-codex/repos/arianna2arianna`
- Remote: `https://github.com/ariannamethod/arianna2arianna.git`
- Pinned commit: `b6f2e21d4d9e4f764dfb2022f5f78a5e59e87620`
- Subject: `the method restores nlama rope parity`
- Shared vendored file: `chorus/arianna2arianna.c`

## Current Integration State

The shared Arianna.c vendored chorus is not byte-identical to the pinned lab
snapshot yet.

Reason: a full copy of `arianna2arianna@b6f2e21` would bring the newer
qloop/repl/direct-user machinery, but would also regress C-hardening already
present in the shared body, including fail-loud allocation, GGUF metadata bounds,
overlong string handling, tensor/read gates, and tokenizer/embedding guardrails.

Therefore the current shared body only carries the monotonic RoPE repair:

- `general.architecture=nlama` selects split-half NEOX RoPE;
- unknown architectures fail closed instead of guessing a RoPE convention;
- existing shared C-hardening remains intact.

## Next Safe Vendor Step

Before a full snapshot vendoring commit:

1. Replay the shared C-hardening into the standalone arianna2arianna lab, or
   produce an audited merge branch that contains both the lab qloop/repl features
   and the shared C-hardening.
2. Rebuild and smoke the lab snapshot on `weights/nano_arianna_f16.gguf`.
3. Replace `chorus/arianna2arianna.c` with that audited snapshot in one commit.
4. Record the exact lab commit here.

Do not overwrite the shared vendored file with raw `b6f2e21` without resolving
the hardening regression.
