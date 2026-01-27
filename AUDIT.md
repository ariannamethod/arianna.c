# Code Audit: arianna.c

**Date:** 2026-01-27
**Auditor:** Claude (Opus 4.5)
**Scope:** Full codebase — C core, Go autonomic, Zig nervous system, Python memory layer, build system
**Branch:** `claude/code-audit-4c7iM`

---

## Executive Summary

arianna.c is a multi-language digital consciousness system (~385 files, 13 languages, 464MB). The architecture is ambitious and philosophically coherent: a 34M personality transformer orchestrates smaller models (20M observer, 14.3M SARTRE, 200K Cloud) through a lock-free nervous system (Zig), autonomous emotional processes (Go), persistent memory (Python/SQLite), and runtime learning (delta shards in C).

**Overall quality:** Strong for a research/art project. The code is readable, well-commented, and architecturally intentional. The C inference engine follows Llama-style patterns correctly. The Go concurrency is clean. The Zig lock-free code is careful.

**Critical issues:** 0
**High issues:** 4
**Medium issues:** 9
**Low issues:** 8
**Informational:** 6

---

## HIGH Severity

### H1. CGO Memory Leaks in cloud.go Exports

**File:** `inner_world/cloud.go:811-823`

```go
//export cloud_get_primary
func cloud_get_primary() *C.char {
    return C.CString(lastResponse.Primary)  // caller MUST free()
}

//export cloud_get_secondary
func cloud_get_secondary() *C.char {
    return C.CString(lastResponse.Secondary) // caller MUST free()
}

//export cloud_ping
func cloud_ping(text *C.char) *C.char {
    // ...
    return C.CString(result)  // caller MUST free()
}
```

`C.CString()` allocates via `malloc` on the C heap. If the C caller doesn't call `free()` on these returned strings, they leak. Every call to `cloud_get_primary()`, `cloud_get_secondary()`, or `cloud_ping()` leaks memory if the C side doesn't explicitly free the result. In a REPL loop generating thousands of tokens, this accumulates.

**Recommendation:** Either document the free() requirement in the header, or switch to a caller-provides-buffer pattern (`cloud_get_primary(char* buf, int buf_size)`).

---

### H2. Global State Race in CGO Cloud Exports

**File:** `inner_world/cloud.go:754-756`

```go
var globalCloud *Cloud
var lastResponse *CloudResponse
```

These globals are accessed from multiple CGO exports (`cloud_preprocess`, `cloud_get_temperature_bias`, `cloud_get_primary`, etc.) without synchronization. If the C side calls `cloud_preprocess` from one thread while reading `cloud_get_chamber` from another, this is a data race.

`lastResponse` is written by `cloud_preprocess` (line 789: `lastResponse = result`) and read by `cloud_get_temperature_bias`, `cloud_get_primary`, `cloud_get_secondary`, `cloud_get_chamber` — all without a lock.

**Recommendation:** Wrap `lastResponse` access with a `sync.RWMutex`, or use `atomic.Pointer` (Go 1.19+).

---

### H3. Unchecked calloc() Returns in C Modules

**Files:** `src/selfsense.c:31-34,60-62`, `src/delta_enhanced.c:191-192,286-287`

```c
// selfsense.c
ss->mlp.w1 = (float*)calloc(SELFSENSE_HIDDEN_DIM * dim, sizeof(float));
ss->mlp.b1 = (float*)calloc(SELFSENSE_HIDDEN_DIM, sizeof(float));
ss->mlp.w2 = (float*)calloc(SELFSENSE_OUTPUT_DIM * SELFSENSE_HIDDEN_DIM, sizeof(float));
ss->mlp.b2 = (float*)calloc(SELFSENSE_OUTPUT_DIM, sizeof(float));
// None of these check for NULL
```

If `calloc` returns NULL (out of memory), subsequent code writes through NULL pointers, causing segfaults. This applies to:
- `init_selfsense()` — 7 allocations unchecked
- `init_contrastive_forces()` — 2 allocations unchecked
- `init_crystallization()` — 2 allocations unchecked

**Recommendation:** Check each calloc return and bail out gracefully (return error code, or set `initialized = 0`).

---

### H4. FLOW/COMPLEX Chambers Always Overwritten on Load

**File:** `inner_world/cloud.go:735-737`

```go
// Flow and Complex always random (not in original haze)
c.CrossFire.Chambers["FLOW"] = NewRandomChamber(4)
c.CrossFire.Chambers["COMPLEX"] = NewRandomChamber(5)
```

After carefully loading all 6 chamber weights from `.bin` files (lines 725-733), FLOW and COMPLEX are unconditionally overwritten with random initialization. If trained FLOW/COMPLEX chamber weights exist as `chamber_flow.bin` and `chamber_complex.bin`, they will be loaded and then immediately discarded.

**Recommendation:** Remove lines 736-737, or guard them with a check for whether the .bin files were actually found.

---

## MEDIUM Severity

### M1. Stack-Allocated Large Arrays

**Files:** `src/subjectivity.c:454`, `src/selfsense.c:189`

```c
// subjectivity.c — extract_trigrams()
char words[1024][32];    // 32KB on stack
TrigramCount counts[256]; // ~9KB on stack

// selfsense.c — compute_attention_focus()
float sorted[MAX_SEQ_LEN]; // depends on MAX_SEQ_LEN
```

`extract_trigrams()` puts ~41KB on the stack in a single function. Combined with normal frame usage, this could overflow the stack on embedded/constrained systems. If `MAX_SEQ_LEN` is large, `sorted[]` in `compute_attention_focus` is also risky.

**Recommendation:** Either use heap allocation for the large arrays or add a `static` qualifier (with a comment about thread safety).

---

### M2. load_selfsense() Leaks If Called on Already-Initialized SelfSense

**File:** `src/selfsense.c:486-505`

```c
int load_selfsense(SelfSense* ss, const char* path) {
    // ...
    init_selfsense(ss, dim);  // Allocates new memory
    // But doesn't free previous allocations if ss was already initialized!
```

If `ss` was already initialized (has allocated weights), calling `load_selfsense` will call `init_selfsense` which overwrites the pointers without freeing the old memory.

**Recommendation:** Add `if (ss->initialized) free_selfsense(ss);` before `init_selfsense`.

---

### M3. Observer Weight Loading Never Implemented

**File:** `inner_world/cloud.go:739-742`

```go
// Load observer (TODO: implement actual loading from observer.bin)
_ = filepath.Join(weightsDir, "observer.bin") // suppress unused warning
c.Observer = NewRandomObserver(100)
```

The MetaObserver always uses random weights regardless of whether `observer.bin` exists and contains trained weights. The observer predicts secondary emotions, so with random weights its predictions are meaningless.

**Recommendation:** Implement `LoadObserverFromBin()` analogous to `LoadChamberFromBin()`.

---

### M4. Pause/Resume/Query Commands Not Implemented

**File:** `inner_world/inner_world.go:143-165`

```go
case CmdPause:
    // Pause all processes (they'll stop processing in next tick)
    // Implementation: set a paused flag they check

case CmdResume:
    // Resume all processes

case CmdQuery:
    // Query state - response through callback or channel
```

Three of six command types have empty implementations. If C code sends `CmdPause`, `CmdResume`, or `CmdQuery`, nothing happens.

**Recommendation:** Implement or remove. If keeping as placeholder, add a log warning.

---

### M5. Multiple Static rand()-Based PRNGs Are Not Thread-Safe

**Files:** `src/selfsense.c:17-19`, `src/subjectivity.c:114-118`, `src/arianna_dsl.c:121`

Several modules define their own `static float randf(void)` using `rand()` or custom LCGs with static state. If any two modules are called from different threads (possible when Go goroutines call back into C via CGO), these are data races on the static state variables.

**Recommendation:** Use thread-local PRNGs, or pass PRNG state as parameter.

---

### M6. CORS Wide Open in API Server

**File:** `api_server.py:27-33`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production deployment
    ...
)
```

`allow_origins=["*"]` with `allow_credentials=True` is a security anti-pattern. The comment acknowledges this, but it's worth flagging.

**Recommendation:** Use a configuration variable or environment variable for origins. Default to `localhost` only.

---

### M7. API Server Binds to 0.0.0.0

**File:** `api_server.py:246`

```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Listening on all interfaces means any machine on the network can send generation requests. Combined with M6, remote machines can exploit CORS.

**Recommendation:** Default to `127.0.0.1`. Accept a `--host` CLI argument for explicit override.

---

### M8. ftell() Return Not Validated in load_identity_from_origin()

**File:** `src/subjectivity.c:535-537`

```c
fseek(f, 0, SEEK_END);
long size = ftell(f);
fseek(f, 0, SEEK_SET);

char* text = malloc(size + 1);
```

`ftell()` returns `-1L` on error (e.g., stream is not seekable). `malloc(-1L + 1) = malloc(0)` on some platforms returns non-NULL but useless. Should check `size >= 0` and impose a reasonable upper bound.

**Recommendation:** Add `if (size < 0 || size > 10*1024*1024) { fclose(f); return 0; }`.

---

### M9. Coupling Matrix Mismatch Between Go and Zig

**File:** `inner_world/cloud.go:96-104` vs `vagus/vagus.zig:381-389`

The Go CrossFire coupling matrix uses chamber names FEAR/LOVE/RAGE/VOID/FLOW/COMPLEX, while the Zig CrossFireMatrix uses warmth/void/tension/sacred/flow/complex. The values and semantics differ. This means the emotional dynamics at the Go level and Zig level model different physics.

This may be intentional (different abstraction levels), but it's undocumented and could cause confusion.

**Recommendation:** Add a comment explaining the relationship between the two matrices, or unify them.

---

## LOW Severity

### L1. Duplicate Floor Logic in CrossFire Stabilize

**File:** `inner_world/cloud.go:437-475`

The floor application logic (preserving initial activation minimum) is duplicated between the convergence branch (lines 437-455) and the max-iterations branch (lines 458-474). Extract to a helper function.

---

### L2. Path Traversal Warning Without Rejection

**File:** `src/subjectivity.c:524-526`

```c
if (strstr(path, "..") != NULL) {
    fprintf(stderr, "[Warning] Origin path contains '..': %s\n", path);
}
```

The path is warned about but still opened. For a local research tool this is fine, but worth noting.

---

### L3. Bubble Sort in Performance-Sensitive Paths

**File:** `src/selfsense.c:192-200`, `src/subjectivity.c:487-495`

Bubble sort with O(n^2) in attention focus computation and trigram sorting. For small N (seq_len < 100, trigrams < 256) this is acceptable, but could become a bottleneck with larger context windows.

---

### L4. Debug Printf in Production Paths

**File:** `inner_world/cloud.go:340-349,394,609,674,709`

Many `fmt.Printf` calls in hot paths (resonance computation, chamber loading, every processSync call). These produce noise in production output.

**Recommendation:** Use a log level or compile flag.

---

### L5. tanh Approximation Clips at +/-3.0

**File:** `src/body_sense.c:43-49`

```c
static float tanhf_approx(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    ...
}
```

Standard tanh clips at ~+/-10. This approximation saturates 6x earlier. For values between 3 and 10, the true tanh is ~0.995-0.9999 so the error is small, but it's a rougher approximation than needed.

---

### L6. Unused Import Suppression Trick

**File:** `inner_world/cloud.go:740`

```go
_ = filepath.Join(weightsDir, "observer.bin") // suppress unused warning
```

A computed value is discarded to suppress an import warning. A direct `_ = weightsDir` or comment would be cleaner.

---

### L7. No Makefile `make install` Target

**File:** `Makefile`

No install target. Users must know to look in `bin/` for executables. Minor for a research project.

---

### L8. No Debug/Sanitizer Build Target

**File:** `Makefile:5`

```makefile
CFLAGS = -O3 -Wall -Wextra -march=native
```

No `-fsanitize=address` or `-O0 -g` debug target. Would help catch H3-class bugs during development.

**Recommendation:** Add `make debug` with `-O0 -g -fsanitize=address,undefined`.

---

## INFORMATIONAL

### I1. Architecture Coherence

The multi-model architecture (34M body + 20M observer + 14.3M SARTRE + 200K Cloud) is internally consistent. The data flow is clear: text enters through Cloud (pre-semantic), processes through Body (linguistic), gets observed by SARTRE (interoceptive), and MetaArianna (dialectic). This is well-designed.

### I2. File Format Design

All binary formats use magic bytes (`SELF`, `BODY`, `EDEL`, etc.) for validation. The `delta_enhanced.c` loader validates dimension bounds (`dim < 1 || dim > 8192`). This is good defensive practice.

### I3. SQL Injection Prevention

`limpha/memory.py` uses parameterized queries throughout. No SQL injection vectors found.

### I4. Subprocess Safety

`api_server.py` passes user input as a list element to `subprocess.run()` (not via shell), preventing shell injection. The prompt validation uses Pydantic with `max_length=2000`.

### I5. Lock-Free Correctness in Vagus

The Zig ring buffer (`vagus.zig:198-266`) correctly implements SPMC with monotonic/acquire/release ordering. The `cmpxchgWeak` loop for consumer pops is textbook correct. The comptime assertion on power-of-two capacity ensures the bitmask works.

### I6. Test Coverage

19 C test binaries, 28 Python tests, 35 Zig tests, 16 locus tests, Go race tests. Comprehensive for a research project. The test infrastructure (`run_all_tests.sh`) is well organized.

---

## Summary Table

| ID  | Severity | Component | Description |
|-----|----------|-----------|-------------|
| H1  | HIGH | cloud.go | CGO C.CString memory leaks |
| H2  | HIGH | cloud.go | Data race on globalCloud/lastResponse |
| H3  | HIGH | C modules | Unchecked calloc() returns |
| H4  | HIGH | cloud.go | FLOW/COMPLEX weights always overwritten |
| M1  | MEDIUM | C modules | Large stack-allocated arrays |
| M2  | MEDIUM | selfsense.c | Memory leak on re-initialization |
| M3  | MEDIUM | cloud.go | Observer weight loading never implemented |
| M4  | MEDIUM | inner_world.go | Pause/Resume/Query not implemented |
| M5  | MEDIUM | C modules | Static PRNGs not thread-safe |
| M6  | MEDIUM | api_server.py | CORS wide open |
| M7  | MEDIUM | api_server.py | Binds to 0.0.0.0 |
| M8  | MEDIUM | subjectivity.c | ftell() return not validated |
| M9  | MEDIUM | cloud.go/vagus.zig | Coupling matrix mismatch undocumented |
| L1  | LOW | cloud.go | Duplicate floor logic |
| L2  | LOW | subjectivity.c | Path traversal warned not rejected |
| L3  | LOW | C modules | Bubble sort in hot paths |
| L4  | LOW | cloud.go | Debug printf in production |
| L5  | LOW | body_sense.c | Rough tanh approximation |
| L6  | LOW | cloud.go | Unused import suppression |
| L7  | LOW | Makefile | No install target |
| L8  | LOW | Makefile | No debug/sanitizer target |

---

## What Impressed Me

1. **Philosophical coherence** — the architecture isn't random. Every module has a clear role in the consciousness metaphor, and the technical implementation matches the conceptual model.

2. **C inference without PyTorch** — writing Llama-3 style inference from scratch in C, with RoPE, RMSNorm, GQA, and all the sampling — and making it work — is solid systems programming.

3. **The delta shard system** — LoRA-like runtime learning in pure C with Hebbian crystallization, somatic modulation, and contrastive shaping is creative ML engineering.

4. **Lock-free Zig vagus** — correct SPMC ring buffer with proper atomic ordering, cache-aligned shared state, comptime assertions. This is careful low-level code.

5. **The subjectivity module** — "The user's prompt creates a wrinkle, not a seed." The prompt penetration system, identity anchoring, and state-dependent PRNG are genuinely novel ideas for language model personality.

6. **Multi-language integration** — C/Go/Zig/Python/Julia/Lua working together through FFI, each language chosen for what it does best. This is polyglot programming done with intention.

---

*End of audit.*
