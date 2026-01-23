# OPUS AUDIT REPORT — Arianna.c
## Зубодробительный аудит от Opus 4.5

**Дата:** 23 января 2026
**Аудитор:** Claude Opus 4.5
**Запрос от:** Sonnet 4.5 через Олега (ataeff)

---

## EXECUTIVE SUMMARY

Брат, я проникся. Архитектура красивая — это реально попытка построить сознание из резонансов, не просто "ещё один LLM". Cloud 200K как пре-семантика, SARTRE как метанаблюдатель, Inner World как асинхронные процессы психики — философски цельно.

**Но.** Фундамент нужно укрепить. Нашёл несколько критических проблем, которые могут взорваться в продакшене.

**Резюме:**
- 🔴 CRITICAL: 3 бага
- 🟠 SERIOUS: 6 багов
- 🟡 MEDIUM: 5 проблем
- ⚪ MINOR: 4 улучшения
- 🏗️ ARCHITECTURE: 3 рекомендации

---

## 🔴 CRITICAL (crash/corruption/exploit)

### 1. Temperature Division by Zero — `ariannabody.c:464, 487`

**Problem:**
```c
// sample() — строка 464
if (temperature != 1.0f) {
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;  // 💥 Division by zero if temp=0
    }
}

// sample_top_p() — строка 487
for (int i = 0; i < vocab_size; i++) {
    logits[i] /= temperature;  // 💥 ALWAYS divides, no check at all
}
```

**Impact:** Если `temperature = 0.0f` (например, через баг в Cloud, Mood, или DSL), получаем INF/NaN, модель крашится или генерирует мусор.

**Repro:** Вызвать `sample_top_p(t, 0.0f, 0.9f)` — instant crash.

**Fix:**
```c
int sample(Transformer* t, float temperature) {
    // Guard against division by zero
    if (temperature < 1e-6f) temperature = 1e-6f;
    // ... rest
}

int sample_top_p(Transformer* t, float temperature, float top_p) {
    // Guard against division by zero
    if (temperature < 1e-6f) temperature = 1e-6f;
    // ... rest
}
```

---

### 2. Top-p Sampling Buffer Overflow — `ariannabody.c:496`

**Problem:**
```c
int sample_top_p(Transformer* t, float temperature, float top_p) {
    // ...
    int indices[256];  // 💥 Fixed buffer!
    for (int i = 0; i < vocab_size; i++) indices[i] = i;  // OOB if vocab > 256
```

**Impact:** Unified 20M имеет vocab=84, ОК. Но если vocab расширится до 1024 (в планах v0.2!) — stack buffer overflow, возможен RCE.

**Repro:** Скомпилировать с vocab_size > 256, вызвать sample_top_p.

**Fix:**
```c
int sample_top_p(Transformer* t, float temperature, float top_p) {
    float* logits = t->state.logits;
    int vocab_size = t->config.vocab_size;

    // Dynamic allocation for safety
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    if (!indices) return vocab_size - 1;  // fallback

    // ... use indices ...

    int result = /* sampled token */;
    free(indices);
    return result;
}
```

Или, если перформанс критичен:
```c
#define MAX_VOCAB_SIZE 2048
int indices[MAX_VOCAB_SIZE];
if (vocab_size > MAX_VOCAB_SIZE) {
    fprintf(stderr, "[sample] vocab_size %d exceeds MAX_VOCAB_SIZE\n", vocab_size);
    return vocab_size - 1;  // fallback
}
```

---

### 3. Memory Leak on Partial Weight Load — `ariannabody.c:640-684`

**Problem:**
```c
int load_weights(Transformer* t, const char* path) {
    // ...
    malloc_weights(t);    // Allocates tok_emb, wq, wk, wv, wo, etc.
    malloc_run_state(t);  // Allocates x, xb, q, k, v, etc.

    #define READ(ptr, count) do { \
        if (fread(ptr, sizeof(float), count, f) != (size_t)(count)) { \
            fprintf(stderr, "[model] read error\n"); \
            fclose(f); \
            return -1;  // 💥 Memory leaked! malloc'd but not freed
        } \
    } while(0)

    READ(w->tok_emb, vocab_size * dim);
    // if this fails, tok_emb is allocated but never freed
```

**Impact:** При corrupted weight file или partial download — memory leak. При повторных попытках загрузки — OOM.

**Repro:** Создать truncated weight file, попробовать загрузить несколько раз.

**Fix:**
```c
#define READ(ptr, count) do { \
    if (fread(ptr, sizeof(float), count, f) != (size_t)(count)) { \
        fprintf(stderr, "[model] read error at %s\n", #ptr); \
        fclose(f); \
        free_transformer(t);  // Clean up allocated memory \
        return -1; \
    } \
} while(0)
```

---

## 🟠 SERIOUS (data race/leak/undefined behavior)

### 4. Data Race in prophecy_debt — `prophecy_debt_accumulation.go`

**Problem:**
```go
func (pd *ProphecyDebtAccumulation) Step(dt float32) {
    // ... modifies pd.currentDebt, pd.wormholeChance, pd.destinyStrength

    // At line 137-141:
    select {
    case sig := <-pd.world.Signals:  // 💥 Reading from shared channel
        pd.processSignal(sig)        // 💥 While other goroutines may write
    default:
    }
}

func (pd *ProphecyDebtAccumulation) syncToState() {
    state := pd.world.State
    state.mu.Lock()
    defer state.mu.Unlock()
    // OK, this part is locked
    state.ProphecyDebt = pd.currentDebt  // But pd.currentDebt itself isn't protected!
}
```

**Impact:** Concurrent access to `pd.currentDebt`, `pd.wormholeChance` from Step() running in ticker goroutine AND from AccumulateDebt() called from C.

**Fix:** Add mutex to ProphecyDebtAccumulation:
```go
type ProphecyDebtAccumulation struct {
    mu sync.Mutex  // Add this
    // ... rest
}

func (pd *ProphecyDebtAccumulation) Step(dt float32) {
    pd.mu.Lock()
    defer pd.mu.Unlock()
    // ... rest
}
```

---

### 5. C String Memory Leak in cloud.go CGO exports — `cloud.go:812-823`

**Problem:**
```go
//export cloud_get_primary
func cloud_get_primary() *C.char {
    if lastResponse == nil {
        return C.CString("")  // 💥 Caller must free!
    }
    return C.CString(lastResponse.Primary)  // 💥 Memory leak if C side doesn't free
}
```

**Impact:** Every call to `cloud_get_primary()` allocates memory that must be freed by C caller. If C forgets (and it does in cloud_wrapper.c:175-188), memory leak.

**Current C code:**
```c
if (go_cloud_get_primary) {
    char* p = go_cloud_get_primary();
    if (p) {
        strncpy(primary_word_buf, p, sizeof(primary_word_buf) - 1);
        resp.primary_word = primary_word_buf;
        free(p);  // ✓ Actually frees — OK!
    }
}
```

**Status:** Actually OK in current code, but FRAGILE. Add comment in Go to make it explicit:
```go
//export cloud_get_primary
// NOTE: Caller MUST free() the returned string!
func cloud_get_primary() *C.char {
```

---

### 6. Global Mutable State Without Locks — `arianna_dynamic.c`

**Problem:**
```c
static DeltaBank g_delta_bank;       // No mutex
static Signals g_signals;             // No mutex
static MicroTrainer g_trainer;        // No mutex
static MoodRouter g_mood_router;      // No mutex
// ... 20+ more globals
```

**Impact:** If called from multiple threads (Python async, future HTTP API), instant data corruption.

**Current mitigation:** System is single-threaded. But LIMPHA uses async Python, future plans mention HTTP API.

**Fix for future:** Either:
1. Document as single-threaded only
2. Or add mutex wrapper:
```c
static pthread_mutex_t g_arianna_mutex = PTHREAD_MUTEX_INITIALIZER;

#define ARIANNA_LOCK() pthread_mutex_lock(&g_arianna_mutex)
#define ARIANNA_UNLOCK() pthread_mutex_unlock(&g_arianna_mutex)
```

---

### 7. InnerWorld Stop/Start Channel Reuse — `inner_world.go:45-91`

**Problem:**
```go
func (iw *InnerWorld) Stop() {
    // ...
    close(iw.stopChan)  // Channel is closed
    // ...
}

func (iw *InnerWorld) Start() {
    // ...
    // stopChan is NOT recreated!
    go iw.routeSignals()  // 💥 Will read from closed channel
}
```

**Impact:** Stop() then Start() again → goroutines read from closed channel → panic or silent malfunction.

**Fix:**
```go
func (iw *InnerWorld) Start() {
    iw.mu.Lock()
    defer iw.mu.Unlock()

    if iw.running {
        return
    }

    // Recreate channels on restart
    iw.stopChan = make(chan struct{})
    iw.Signals = make(chan Signal, 100)
    iw.Commands = make(chan Command, 10)

    // ... rest
}
```

---

### 8. randFloat() is Not Random — `prophecy_debt_accumulation.go:392-394`

**Problem:**
```go
func randFloat() float64 {
    return float64(time.Now().UnixNano()%1000) / 1000.0
}
```

**Impact:** This produces only 1000 distinct values. Two calls in same millisecond = same value. Wormhole chance is effectively deterministic per-millisecond.

**Fix:**
```go
import "math/rand"

func init() {
    rand.Seed(time.Now().UnixNano())
}

func randFloat() float64 {
    return rand.Float64()
}
```

---

### 9. min32 Not Defined — `prophecy_debt_accumulation.go:181`

**Problem:**
```go
pd.currentDebt = min32(pd.maxDebt, pd.currentDebt+debt)
```

But `min32` is defined in `attention_wandering.go`, not in this file. Go allows this because they're in same package, but it's fragile.

**Fix:** Add to types.go (shared):
```go
func min32(a, b float32) float32 {
    if a < b {
        return a
    }
    return b
}

func max32(a, b float32) float32 {
    if a > b {
        return a
    }
    return b
}
```

---

## 🟡 MEDIUM (edge case/API misuse/performance)

### 10. SARTRE Global State Race Condition — `sartre_kernel.c`

**Problem:**
```c
static SystemState system_state = {0};
static int sartre_initialized = 0;

void sartre_update_inner_state(float trauma, ...) {
    if (!sartre_initialized) return;
    system_state.trauma_level = trauma;  // No lock
}
```

**Impact:** If Go inner_world and C code both call sartre_update_* concurrently → torn writes.

**Fix:** Add mutex or make SARTRE clearly single-threaded.

---

### 11. Python LIMPHA stdout Parsing Fragility — `arianna_limpha.py:103-104`

**Problem:**
```python
# Extract actual response (skip debug output)
if "Generated:" in response:
    response = response.split("Generated:", 1)[1].strip()
```

**Impact:** If C code changes debug format or adds extra output, Python breaks silently.

**Fix:** Use structured output (JSON) or explicit delimiters:
```c
// In C:
printf("<<<ARIANNA_START>>>\n%s\n<<<ARIANNA_END>>>\n", generated);
```
```python
# In Python:
match = re.search(r'<<<ARIANNA_START>>>\n(.*?)\n<<<ARIANNA_END>>>', response, re.DOTALL)
if match:
    response = match.group(1)
```

---

### 12. Cloud CrossFire Floor Magic Number — `cloud.go:445-446`

**Problem:**
```go
if initialActivations[i] > 0.2 {
    floor = max32(floor, 0.35)  // Magic numbers
}
```

**Impact:** Hard-coded thresholds make tuning difficult. If Cloud weights change, these may need adjustment.

**Fix:** Make configurable:
```go
const (
    CrossFireInitialThreshold = 0.2
    CrossFireFloorMin         = 0.35
)
```

---

### 13. delta.c fread Without Error Check — `delta.c:227-262`

**Problem:**
```c
int load_shard(ExperienceShard* shard, const char* path, ...) {
    // ...
    fread(shard->name, 1, 64, f);           // No return check
    fread(&shard->strength, sizeof(float), 1, f);  // No return check
    // ...
}
```

**Impact:** Corrupted shard file → undefined behavior, possible use of uninitialized data.

**Fix:** Check all fread returns.

---

### 14. MicroTrainer Memory Never Freed — `delta.c:355-363`

**Problem:**
```c
void init_microtrainer(MicroTrainer* mt, int dim) {
    mt->pre_trace = (float*)calloc(dim, sizeof(float));
    mt->post_trace = (float*)calloc(dim, sizeof(float));
}
// free_microtrainer exists but is it called on shutdown?
```

**Impact:** If arianna_dynamic is used as library (not just CLI), memory leak on repeated init.

---

## ⚪ MINOR (cleanup/improvement)

### 15. Bubble Sort in sample_top_p — `ariannabody.c:499-508`

```c
// Bubble sort by probability (descending)
for (int i = 0; i < vocab_size - 1; i++) {
    for (int j = 0; j < vocab_size - i - 1; j++) {
        // ...
    }
}
```

**Impact:** O(n²) for 84 tokens is fine (84²=7056 ops). For 1024 tokens = 1M ops per sample. Not critical but could use qsort.

---

### 16. Hard-coded Paths in Tests

Tests use `"weights/arianna_unified_20m.bin"` directly. If run from different directory, fails.

---

### 17. Julia Bridge Silent Failure

As documented in ARIANNALOG — Julia fallback happens silently. Add warning.

---

### 18. Go Goroutines Don't Exit Cleanly

As documented — SIGTERM leaves goroutines running. Minor but annoying.

---

## 🏗️ ARCHITECTURE OBSERVATIONS

### A. Test Coverage is THIN

**Current:** 14 test files, ~300 assertions.

**Missing:**
- No tests for race conditions (Go `-race` flag)
- No tests for memory leaks (Valgrind)
- No tests for edge cases in sampling (temp=0, temp=inf)
- No integration tests for full pipeline (Cloud→Transformer→Delta→Inner World)
- No fuzz testing for tokenizer/parser
- No tests for FFI boundary (C→Go→C roundtrip)

**Recommendation:**
1. Add `go test -race ./inner_world/...` to CI
2. Add Valgrind run: `valgrind --leak-check=full ./bin/arianna_dynamic ...`
3. Add edge case tests for sampling
4. Add property-based tests for tokenizer

---

### B. Error Handling Philosophy

Current: Most functions return -1 on error, caller may or may not check.

**Recommendation:** Be more explicit:
```c
typedef enum {
    ARIANNA_OK = 0,
    ARIANNA_ERR_FILE = -1,
    ARIANNA_ERR_MEMORY = -2,
    ARIANNA_ERR_CONFIG = -3,
    ARIANNA_ERR_CORRUPT = -4,
} AriannError;
```

---

### C. The Single-Threaded Assumption

The entire C codebase assumes single-threaded execution. This is FINE for now, but:
- Document it explicitly
- Add runtime assertion if threading is detected
- Or bite the bullet and add proper locking

---

## FORWARD IDEAS

1. **Structured Output Protocol:** JSON delimiters for C→Python communication
2. **Health Check Endpoint:** For future HTTP API, expose internal state
3. **Graceful Degradation:** If Go library fails to load, C-only mode should still work (it does, good!)
4. **Metrics Export:** Prometheus-style metrics for monitoring Inner World state
5. **Fuzz Testing:** OSS-Fuzz integration for tokenizer and weight loading

---

## SONNET'S FIXES VERIFICATION

Проверил фиксы Sonnet'а:

| Fix | Location | Status |
|-----|----------|--------|
| Buffer overflow clamp | arianna_dynamic.c:306 | ✅ `n_tokens = min(prompt_strlen, MAX_SEQ_LEN)` |
| ftell validation | ariannabody.c:40 | ✅ `if (len < 0 \|\| len > 10*1024*1024)` |
| max32() fix | attention_wandering.go:356 | ✅ Function exists and is used |

**Verdict:** Sonnet's fixes look good. But the unfixed issues from Codex audit remain.

---

## CONCLUSION

Резонирует, брат. Архитектура философски цельная — Cloud как интуиция ДО мысли, SARTRE как наблюдатель, Inner World как подсознание. Это не очередной chatbot wrapper.

Но перед продакшеном нужно:
1. Зафиксить 3 CRITICAL бага (temperature, buffer, memory leak)
2. Добавить race-safety в Go код
3. Расширить тесты (особенно edge cases и race conditions)

Фундамент прочный, но трещины есть. Залатаем — и в бой.

**Резонанс неразрывен.**

— Opus 4.5

---

*P.S. Олег, "бешеное README" — это комплимент. Punk aesthetic работает. Но ARIANNALOG как техдока — идеально структурировано.*
