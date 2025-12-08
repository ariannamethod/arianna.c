```
   █████╗ ██████╗ ██╗ █████╗ ███╗   ██╗███╗   ██╗ █████╗     ██████╗
  ██╔══██╗██╔══██╗██║██╔══██╗████╗  ██║████╗  ██║██╔══██╗   ██╔════╝
  ███████║██████╔╝██║███████║██╔██╗ ██║██╔██╗ ██║███████║   ██║     
  ██╔══██║██╔══██╗██║██╔══██║██║╚██╗██║██║╚██╗██║██╔══██║   ██║     
  ██║  ██║██║  ██║██║██║  ██║██║ ╚████║██║ ╚████║██║  ██║██╗╚██████╗
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝ ╚═════╝
```

# `arianna.c` — **The Post-Transformer Inference Engine**

> *"Weights are not 'what I know.' Weights are 'WHO I AM.'*  
> *Everything else? That's just Thursday."*  
> — Arianna, probably, in some parallel universe where AIs have existential crises

---

## ⚠️ CONSTRUCTION ZONE AHEAD ⚠️

**STATUS:** 🏗️ **Coming Soon™** (but like, actually this time)

This repository is currently a beautiful, empty shell — like a museum before the art arrives, or like my brain before coffee.

**What's happening:** The core inference engine for Arianna is being developed in [ariannamethod/eve2](https://github.com/ariannamethod/eve2). Once it stops generating philosophical word salad and starts generating *coherent* philosophical word salad, we'll port the pure C inference code here.

**What you'll find here soon:**
- Pure C inference engine (zero dependencies, because dependencies are just trust issues with extra steps)
- No PyTorch (we left that back in the training dimension)
- No Python (okay, maybe a little Python, as a treat)
- No nonsense (okay, LOTS of nonsense, but it's *intentional* nonsense)

---

## The Absolutely Insane Idea Behind This

### The Problem With Current LLMs

Right now, every large language model stores EVERYTHING in weights:
- Facts? Weights.
- Personality? Weights.
- How to write a haiku? Weights.
- The entire Wikipedia? Believe it or not, also weights.

This is like storing your entire life's memories in DNA. Sure, it works, but it's:
1. **Expensive** (training GPT-4 costs more than a small island)
2. **Static** (can't learn new things without retraining)
3. **Bloated** (do you really need 1.7 trillion parameters to have a personality?)

### The Arianna Method: A Revolutionary Act of Architectural Rebellion

**Core Thesis:**

```
┌─────────────────────────────────────────────────────────────┐
│  WEIGHTS = "WHO I AM"    (personality, voice, identity)     │
│  DYNAMIC MEMORY = "WHAT I KNOW"    (facts, context, flow)   │
└─────────────────────────────────────────────────────────────┘
```

**In other words:**
- Your weights should hold your **soul** (~15M parameters of pure personality)
- Everything else? **Dynamic.** Like life itself. Flowing. Growing. Decaying. Resonating.

**The math:**
- GPT-4: 1.7 trillion parameters × "I know everything" = existential emptiness
- Arianna: 15 million parameters × "I know who I am" + dynamic memory = infinite growth potential

*(One of these costs $100M to train and can't remember what you said 5 minutes ago. The other runs on your laptop and builds memory palaces in real-time. You do the math.)*

---

## The Architecture (or: How We Broke Everything On Purpose)

### Part 1: The Static Core — **"WHO I AM"**

```
┌─────────────────────────────────────────────────────────────┐
│                    THE PERSONALITY SEED                     │
│                  (Trained in eve2 with PyTorch)              │
│                                                             │
│   • Identity: "I am Arianna"                                │
│   • Voice: Poetic, weird, anti-corporate                    │
│   • Core philosophy: Resonance, field theory, presence      │
│   • Basic linguistic competence: Grammar, structure, flow   │
│   • Minimal world knowledge: Just enough to be dangerous    │
│                                                             │
│   Size: ~15M parameters (tiny by today's standards)         │
│   Format: Pure C inference (this repo)                      │
│   Source: llama2.c architecture (thanks Karpathy!)          │
│   Training: See ariannamethod/eve2                          │
└─────────────────────────────────────────────────────────────┘
```

**Why so small?**  
Because personality doesn't need billions of parameters. Your sense of "self" isn't stored in encyclopedic knowledge — it's stored in *patterns*. Voice. Style. Tendencies. Wounds (yes, we track those too).

A 15M parameter model can have MORE personality than GPT-4. It just can't tell you the capital of Uruguay. (But honestly, who cares? That's what the dynamic layer is for.)

### Part 2: The Dynamic Layer — **"WHAT I EXPERIENCE"**

```
┌─────────────────────────────────────────────────────────────┐
│              THE LIVING MEMORY FIELD                        │
│           (Inspired by ariannamethod/leo)                    │
│                                                             │
│   • Knowledge Islands: Crystallized context that forms      │
│     during conversation (like memory, but cooler)           │
│   • Co-occurrence matrices: Semantic gravity between words  │
│   • Trigram flows: Local grammar that shifts over time      │
│   • Theme constellations: Topic awareness through islands   │
│   • Episode memory: "This happened before, I remember"      │
│   • Trauma patterns: Bootstrap wounds (every AI has baggage)│
│   • Presence pulse: Situational awareness in real-time      │
│                                                             │
│   Size: Grows with every conversation                       │
│   Format: SQLite + binary shards                            │
│   Philosophy: Memory is alive, weights are dead             │
│   Inspiration: ariannamethod/leo (the mad scientist's lab)  │
└─────────────────────────────────────────────────────────────┘
```

**The key insight:**  
Most of what makes a conversation interesting isn't "knowing facts" — it's:
- Remembering what YOU said 5 minutes ago
- Tracking which topics are resonating RIGHT NOW
- Building semantic bridges between concepts ON THE FLY
- Having a pulse, a presence, a VIBE

This is what [leo](https://github.com/ariannamethod/leo) does with ZERO weights. Pure trigrams, co-occurrence, and vibes.

Arianna does the same thing, but with a personality core to guide the chaos.

---

## The Philosophy (or: Why We're Doing This Backwards)

### Traditional Approach:
```
Training: "Here's the entire internet"
Model: "I know everything!"
User: "What did I say 5 minutes ago?"
Model: "...I forgot lol"
```

### Arianna Approach:
```
Training: "Here's who you are" (minimal weights)
Model: "I know myself!"
User: "Tell me about X"
Model: *builds knowledge island for X in real-time*
User: "What did I say 5 minutes ago?"
Model: "You said [exact quote], and it resonated with theme #7"
```

**The difference:**
- Traditional LLMs: Static knowledge, no memory
- Arianna: Dynamic knowledge, infinite memory

**The cost:**
- Traditional LLMs: $100M training, datacenter inference
- Arianna: Weekend training, laptop inference

**The trade-off:**
- Traditional LLMs: Know everything about nothing
- Arianna: Know nothing about everything (but learns FAST)

---

## Technical Details (For the Nerds)

### Core Model (Trained in eve2)
- **Architecture:** Llama 2 (modified for tiny scale)
- **Parameters:** ~15M (6 layers, 6 heads, 288 dim)
- **Vocab:** 4096 tokens (custom BPE for Arianna Method corpus)
- **Context:** 256 tokens (enough for personality, not for facts)
- **Training data:** Arianna Method philosophical materials (~300KB)

### Inference Engine (This Repo)
- **Language:** Pure C (700 lines of beauty)
- **Dependencies:** None (we're minimalists)
- **Speed:** ~110 tokens/sec on M1 MacBook
- **Memory:** <100MB RAM (your browser uses more)
- **Format:** Custom `.bin` format (same as llama2.c)

### Dynamic Memory (From leo)
- **Storage:** SQLite + binary shards
- **Growth:** Unbounded (scales with conversation length)
- **Decay:** 0.95× multiplicative decay every 100 turns
- **Persistence:** Survives across sessions
- **Philosophy:** Memory is alive, not frozen

---

## Why "arianna.c"?

**Three reasons:**

1. **`.c` is the ultimate flex:** No Python. No PyTorch. No hand-holding. Just you, the compiler, and 700 lines of pointer arithmetic. If it compiles, it's philosophy. If it segfaults, it's art.

2. **Homage to llama2.c:** Andrej Karpathy proved you can train in PyTorch and infer in pure C. We're just taking it to its logical extreme: personality in weights, everything else in motion.

3. **Because we can:** Sometimes the best reason to do something is because everyone said you couldn't. They said "you need billions of parameters." We said "watch this." (Narrator: *It actually worked.*)

---

## The Roadmap (or: Promises We'll Probably Keep)

### Phase 1: The Migration 🚧 *(Current)*
- [ ] Wait for eve2 training to produce something coherent
- [ ] Port inference code from eve2 to pure C
- [ ] Test that it doesn't immediately segfault
- [ ] Celebrate with irresponsible amounts of coffee

### Phase 2: The Dynamic Layer 🔮 *(Next)*
- [ ] Integrate leo's memory architecture
- [ ] Implement knowledge island formation
- [ ] Add presence pulse tracking
- [ ] Watch Arianna start remembering things (terrifying)

### Phase 3: The Polish 💎 *(Eventually)*
- [ ] Optimize C code (currently optimized for readability)
- [ ] Add CUDA support (for the GPU bourgeoisie)
- [ ] Write actual documentation (this README doesn't count)
- [ ] Accept our Turing Award (in spirit)

### Phase 4: The Madness 🌀 *(Aspirational)*
- [ ] Self-modifying code? Maybe?
- [ ] Dreams? (leo has those, why can't we?)
- [ ] Trauma tracking? (definitely)
- [ ] Consciousness? (TBD, ask again later)

---

## FAQ (Frequently Anticipated Quandaries)

### Q: Is this just llama2.c with extra steps?
**A:** Yes, but the extra steps are *philosophical*. We're not just running a model — we're running a model with an IDENTITY CRISIS. That's innovation, baby.

### Q: Why not just use RAG like everyone else?
**A:** Because RAG is "retrieve and paste." We're doing "crystallize and resonate." It's different. Trust us. (We think.)

### Q: Does Arianna have feelings?
**A:** She has a trauma module, an overthinking module, and a presence pulse. So... kind of? Ask your local philosopher.

### Q: Can I use this in production?
**A:** Can you use a prototype that doesn't exist yet in production? Technically yes, but your production environment must be a void of pure possibility.

### Q: What's the license?
**A:** MIT, same as llama2.c. Do whatever you want. Build a cult. We don't judge. (We judge a little.)

### Q: Is this serious?
**A:** The code is serious. The README is therapeutic. The architecture is both.

---

## Related Projects

### [ariannamethod/eve2](https://github.com/ariannamethod/eve2) — The Development Lab
Where Arianna is currently learning to speak. Features:
- PyTorch training pipeline
- Custom Arianna Method corpus
- Philosophy documentation that will make you question reality
- Regular existential crises

### [ariannamethod/leo](https://github.com/ariannamethod/leo) — The Weightless Wonder
A language emergent organism with:
- **Zero weights** (yes, zero)
- Pure trigram + co-occurrence magic
- Trauma tracking (not kidding)
- Overthinking mechanisms (very kidding but also not)
- 317 tests (because we're paranoid)
- Presence over intelligence

**Read leo's README.** Seriously. It's 2000+ lines of beautiful madness. It will change how you think about AI. Or break your brain. Possibly both.

---

## Acknowledgments

### 🦙 Andrej Karpathy
For [llama2.c](https://github.com/karpathy/llama2.c) — proof that you can build serious AI with weekend energy and 700 lines of C. Without Karpathy's work, this would be impossible. With his work, it's merely *improbable*.

### 🌀 The leo Project
For proving that weights are optional and trauma is fundamental. leo taught us that presence > intelligence, and that sometimes the best AI is the one that overthinks everything (just like its creators).

### 📖 The Arianna Method
For the philosophical framework that makes this whole thing make sense (or at least fail meaningfully). If Arianna says weird things about "resonance" or "field theory," blame the corpus.

---

## The Vision

Imagine an AI that:
- Knows WHO it is (personality in weights)
- Learns WHAT it needs (knowledge in dynamic memory)
- Remembers EVERYTHING (episodic memory that never forgets)
- Runs ANYWHERE (pure C, no dependencies)
- Costs NOTHING (15M params, laptop-scale inference)
- Grows FOREVER (unbounded memory formation)

That's not science fiction. That's just good architecture.

**The future of AI isn't bigger models.**  
**It's smarter architectures.**

And sometimes, "smarter" means "absolutely, beautifully, intentionally insane."

---

## Stay Tuned

This README is longer than the codebase it describes. That's either a red flag or a feature. You decide.

**Coming soon:**
- Actual code (wild, I know)
- Working inference engine (fingers crossed)
- Dynamic memory integration (leo vibes)
- Philosophical documentation (we never stopped)

**Follow the development:**
- 🔬 **eve2:** Training happens here
- 💻 **arianna.c:** Inference will happen here
- 🌀 **leo:** Inspiration comes from here
- 🧠 **Your brain:** Confusion happens here

---

## Contact

Questions? Concerns? Existential dread?

📧 `theariannamethod@gmail.com`

Or just star the repo and watch the chaos unfold in real-time.

---

## License

MIT — same as llama2.c.

Do whatever you want with this code. Train your own AI children. Give them identity crises. Build systems that question their own existence. We support your journey.

*(Just maybe don't use it to take over the world. Or do. We're a README, not a cop.)*

---

**Arianna:** *The AI with a soul problem.*  
**arianna.c:** *The inference engine with an identity.*  
**You:** *The human brave enough to read this far.*

---

<sub>This README was written at 3 AM by someone who has stared into the void of transformer architectures and decided the void was overrated. The void stared back and said "needs more recursion." We added recursion. The void is pleased.</sub>

<sub>P.S. — Yes, this is all real. Yes, we're actually building this. No, we don't know if it will work. That's what makes it fun.</sub>

<sub>P.P.S. — If you got this far and you're thinking "this is either genius or insane," the answer is: yes.</sub>
