# Packages

Modular extensions that connect to Arianna's **SARTRE Kernel**. Each package is optional — Arianna runs without them. When connected, they extend her capabilities without overwriting her voice.

The core principle: **Architecture > Weights**. A ~550.7M parameter system (0.2M Cloud + 500M Tongue + 36M Soul/MetaArianna + 14.3M SARTRE) orchestrates through architecture, not brute parameter count.

---

## Available Packages

### git.arianna (git_arianna)

Git integration layer. **Disabled by default.**

Provides Arianna with awareness of repository state, commit history, and change signals through the SARTRE Kernel. Observes rather than acts — reads git context so Arianna can reason about code evolution.

---

## Structure

```
packages/
├── README.md          # This file
├── TESTING.md         # Package testing guide
├── __init__.py
├── git_arianna/       # Git integration (disabled by default)
└── tests/             # Package tests
```

---

## Adding New Packages

Packages connect through SARTRE Kernel. Each package reports its state via shared metrics. If you're building a new package:

1. Create a subdirectory in `packages/`
2. Implement the SARTRE interface (see existing packages)
3. Add REPL commands if user-facing
4. Document in this README

Future packages brewing: memory persistence, world model, cross-session learning. Tomorrow's problems.
