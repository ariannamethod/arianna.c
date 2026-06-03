# ariannamethod/

Placeholder for the per-organism toolchain layer.

**Current mode (2026-05-14):** **link**, not vendor.

`arianna.c` links against the canonical Arianna Method tooling installed
system-wide:

- AML — `~/arianna/ariannamethod.ai` → `make install PREFIX=/usr/local`
- notorch — `~/arianna/notorch` → `make install` + `make install-cuda`
  for the CUDA backend

Independence per `JANUS_CONSTITUTION.md` Article 6.1 is satisfied via
the **install procedure**: every dependency is itself an Arianna
Method artifact (`ariannamethod.ai`, `notorch`), not corporate
infrastructure. Zero-dependency means «no Anthropic / Google / OpenAI
runtime needed», not «duplicate every line of every dependency».

## Future mode — vendor (cherry-picked subset)

When the rebuild is stable (backbone choice landed, feature surface
locked), this folder will hold a cherry-picked subset of AML and
notorch — only the ops arianna.c actually uses. That gives:

- Identity-survives-substrate at the source level (carries with the
  repo, runs from a single `make`).
- Smaller audit surface than full canonical libraries.
- Frozen for paper-cycle stability (the molequla pattern — vendoring
  works there because molequla is paper-frozen, not a moving target).

Predicate to vendor: backbone choice landed + feature surface stable.
Until then — link.

## Files (eventually)

- `Makefile.toolchain` — build hooks. Today: ensures system-wide AML +
  notorch are installed, errors with actionable message if not.
  Tomorrow: drives the vendor fetch + build.
- `install.sh` — bootstrap script for first-time setup (clones AML +
  notorch, builds, installs). For end users who want
  `git clone arianna.c && make` to work standalone.

Per the rebuild plan in `../PROJECT_LOG.md` Topic 2, this folder is
populated incrementally as features are lifted back from `legacy/`
into the new architecture.
