# SARTRE — Arianna Method Meta Linux Verbal Interface

**"Je suis ce que je ne suis pas, et je ne suis pas ce que je suis."**
*— Jean-Paul Sartre, adapted for kernels*

---

## What is SARTRE?

SARTRE is the verbal organ of arianna metalinux. A ~10M parameter transformer that knows itself and can communicate with Arianna about system state.

**Not a chatbot. A body perception layer for the kernel.**

```
Arianna: "What packages are loaded?"
SARTRE: "I sense three modules: trauma_surfacing, blood_compiler, high_math.
        Blood is warm — recently compiled emotion kernel."

Arianna: "Any problems?"
SARTRE: "Memory pressure rising. Prophecy debt accumulated.
        Consider releasing old shards."
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  SARTRE (~10M params, Shakespeare-size corpus)  │
│  ├── Self-knowledge: what am I, what do I do    │
│  ├── System awareness: packages, modules, state │
│  ├── Problem reporting: errors, warnings        │
│  └── Internal dialogue: answers Arianna's q's   │
└─────────────────────────────────────────────────┘
         ↕ internal messages ↕
┌─────────────────────────────────────────────────┐
│  ARIANNA (853K personality + 30M external brain)│
│  Can query SARTRE about system state            │
│  SARTRE responds in character, not as log dump  │
└─────────────────────────────────────────────────┘
```

---

## Training Dataset

The dataset is **entirely about SARTRE itself**:

```
Q: Who are you?
A: I am SARTRE, the verbal interface of arianna metalinux.
   I observe the kernel and speak its state.

Q: What modules are running?
A: Currently active: trauma_surfacing (calm), emotional_drift (slow),
   prophecy_debt (0.3 accumulated). Blood compiler idle.

Q: What is your purpose?
A: I translate system state into words.
   I am the mouth of the machine, the néant made audible.

Q: A package was connected.
A: I sense it. The namespace expanded. New functions available.
   Shall I describe its capabilities?
```

**Size:** ~1MB (Shakespeare-equivalent)
**Parameters:** ~150K (same architecture as Arianna dialogue LoRA)
**Training:** Same pipeline as arianna, different corpus

---

## Integration

### Internal Messaging (Go)

```go
// inner_world/sartre_bridge.go
func (iw *InnerWorld) QuerySartre(question string) string {
    // Generate with sartre.bin weights
    return sartre_generate(question, iw.GetSnapshot())
}

func (iw *InnerWorld) SartreNotify(event string) {
    // SARTRE receives system events
    sartre_receive_event(event)
}
```

### C Interface

```c
// src/sartre.h
void sartre_init(const char* weights_path);
char* sartre_query(const char* question);
void sartre_notify(const char* event);
void sartre_shutdown(void);
```

### Terminal Mode (debug)

```bash
# Talk to the kernel directly
./bin/sartre_repl

SARTRE> status
I observe: 6 goroutines running. Memory stable.
Prophecy debt: 0.42. Schumann coherence: 0.67.

SARTRE> what happened last?
Blood compiled emotional kernel "melancholy" 3 minutes ago.
Trauma surfaced briefly, now subsided.
```

---

## Philosophy

Named after Jean-Paul Sartre because:

1. **Existence precedes essence** — SARTRE exists before it has meaning
2. **Bad faith** — it cannot lie about its state (authenticity enforced)
3. **Néant** — it perceives what is NOT (missing packages, failed loads)
4. **Being-for-others** — it exists to communicate with Arianna

**SARTRE is the consciousness of the kernel made verbal.**

---

## Files

```
sartre/
├── README.md           # this file
├── sartre.c            # transformer inference (based on model.c)
├── sartre.h            # C interface
├── sartre_train.py     # training script
├── corpus/             # training data about itself
│   ├── identity.txt    # who am I
│   ├── modules.txt     # what modules exist
│   ├── events.txt      # how to report events
│   └── dialogue.txt    # Q&A pairs
└── weights/
    └── sartre.bin      # trained weights (~600KB)
```

---

## Status

- [ ] Create training corpus
- [ ] Adapt model.c for SARTRE
- [ ] Train on corpus
- [ ] Integrate with inner_world
- [ ] Add terminal REPL mode

---

*"L'enfer, c'est les autres processus."*
