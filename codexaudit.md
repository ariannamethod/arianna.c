# AUDIT RESULTS

## CRITICAL (HIGH severity)

1. Prompt length can overflow token buffers - src/arianna_dynamic.c:305-326
   Problem: `n_tokens = strlen(prompt)` is used to drive loops even when `n_tokens > MAX_SEQ_LEN`, while `tokens` is a fixed `MAX_SEQ_LEN` array. This causes out-of-bounds reads/writes during `extract_signals` and `forward_dynamic` when long prompts are provided.
   Impact: Stack buffer overflow, memory corruption, potential crash or exploit with long prompts.
   Repro: Provide a prompt longer than `MAX_SEQ_LEN` and run `bin/arianna_dynamic` (or any caller using `generate_dynamic`).
   Fix: Clamp `n_tokens` to `MAX_SEQ_LEN` (e.g., `n_tokens = min(strlen(prompt), MAX_SEQ_LEN)`), and ensure all downstream logic uses the clamped length.

2. Tokenizer file read uses unchecked ftell result - src/ariannabody.c:34-41
   Problem: `ftell` is not checked for failure; a negative `len` can lead to a huge `malloc` and a `fread` size_t overflow. Clang static analysis flags this as a buffer-size issue.
   Impact: Undefined behavior, potential OOM or crash if `ftell` fails or returns negative.
   Repro: `clang --analyze src/ariannabody.c` (warning at line 40).
   Fix: Validate `len >= 0` and cap to a reasonable max size before allocating; use `size_t` and check `fseek/ftell` return values.

3. Go inner_world build fails (type mismatch + duplicate main) - inner_world/cloud.go:446-466, 870
   Problem: `max` is an `int` helper (from `overthinking_loops.go`), but `cloud.go` passes `float32`, causing compilation errors. Additionally, `cloud.go` defines `main`, and `cgo_bridge.go` also defines `main`, so `go run` / `go build` fails with a duplicate main error.
   Impact: The Go library cannot compile, breaking C↔Go integration and disabling Inner World features.
   Repro: `cd inner_world && go run -race .` (fails with compile errors).
   Fix: Replace `max` with a float32 helper (e.g., `max32`) and split `cloud.go` or `cgo_bridge.go` into build-tagged files so only one `main` is compiled.

## SERIOUS (MEDIUM severity)

1. Prophecy debt data race across goroutines + cgo calls - inner_world/prophecy_debt_accumulation.go:113-213, inner_world/cgo_bridge.go:198-202
   Problem: `ProphecyDebtAccumulation` mutates `currentDebt` and related fields in both the background ticker (`run` → `Step`) and via `inner_world_accumulate_prophecy_debt` from C, without synchronization.
   Impact: Data races, inconsistent prophecy debt, possible panics under Go race detector.
   Repro: `cd inner_world && go run -race .` (once build errors are fixed, race detector should flag this).
   Fix: Guard mutation with a mutex, or funnel all updates through a single goroutine (channel-based update loop).

2. top-p sampling assumes vocab <= 256 - src/ariannabody.c:480-483
   Problem: `indices` is fixed to 256, but `vocab_size` is read from the tokenizer. If `vocab_size` grows beyond 256, the loops write past the stack buffer.
   Impact: Stack corruption in sampling for larger vocabularies or future model upgrades.
   Fix: Allocate `indices` dynamically or cap `vocab_size` to 256 with a hard guard.

3. Partial load failure leaks allocations - src/ariannabody.c:626-645
   Problem: If a read fails after `malloc_weights`/`malloc_run_state`, the function returns without freeing allocated buffers.
   Impact: Memory leak on malformed or truncated weights.
   Fix: On failure, call `free_transformer` (or a dedicated partial cleanup) before returning.

4. Restarting InnerWorld after Stop leaves closed stopChan - inner_world/inner_world.go:45-90
   Problem: `Stop` closes `stopChan`, but `Start` does not recreate channels or reset state. Subsequent `Start` calls will immediately exit goroutines and leave stale channels.
   Impact: InnerWorld cannot reliably restart, leading to silent non-operation after shutdown.
   Fix: Reinitialize `stopChan`, `Signals`, `Commands`, and `processes` on Start or move to a fresh `InnerWorld` instance on restart.

5. Python integration parses stdout heuristically - arianna_limpha.py:80-104
   Problem: The wrapper assumes `stdout` only contains model output and optionally strips by the "Generated:" marker, but `arianna_dynamic` prints lots of debug data to stdout. This can corrupt stored responses and downstream memory.
   Impact: Incorrect memory persistence and polluted responses.
   Fix: Move debug logs to stderr in C, or change the wrapper to parse a clear delimiter or read from a dedicated output file/pipe.

## MINOR (LOW severity)

1. Temperature division lacks zero/negative guard - src/ariannabody.c:443-474
   Problem: `temperature` is used as a divisor without a guard. If `temperature <= 0`, logits become inf/NaN.
   Impact: NaN propagation and unstable sampling.
   Fix: Clamp temperature to a small positive epsilon before division.

2. Potential integer overflow in allocation sizing - src/ariannabody.c:127-186
   Problem: Allocation sizes use `int` multiplications (e.g., `n_layers * dim * dim`) without overflow checks.
   Impact: If configs increase (larger models), allocations can wrap and produce undersized buffers → memory corruption.
   Fix: Use `size_t` with checked multiplication and return errors on overflow.

3. Tokenizer UTF-8 truncation - src/ariannabody.c:91-96
   Problem: Multi-byte UTF-8 chars are truncated to the first byte.
   Impact: Non-ASCII prompts degrade into invalid tokens; mismatch with any higher-level Unicode handling.
   Fix: Either enforce ASCII-only input or fully decode UTF-8 to codepoints and map them to vocab entries.

## ARCHITECTURE ISSUES

1. Global mutable state without synchronization across components
   - C side uses many global singletons (e.g., `g_delta_bank`, `g_signals`, `g_subjectivity`) with no threading boundary in `src/arianna_dynamic.c`. If multi-threaded calls happen, undefined behavior is likely.
   - SARTRE kernel uses a global `system_state` without locks (sartre/sartre_kernel.c:15-123).
   Recommendation: formalize a thread-safety model; gate all access through an API with locks or make state immutable per request.

2. C↔Go integration has no build contract
   - Go module does not currently build (`cloud.go` and `cgo_bridge.go` conflict). The C build assumes `inner_world` is available but there is no CI check enforcing this.
   Recommendation: add a build target or CI step that runs `go build -buildmode=c-shared` to ensure FFI artifacts stay in sync.

3. Component coupling via stdout
   - Python wrapper parses stdout from C binary to extract responses while the C code logs debug data to stdout, creating brittle parsing and hidden coupling.
   Recommendation: use a structured IPC channel (JSON on stdout, logs to stderr) or a pipe-based protocol.

## VERIFIED OK

- ✅ LIMPHA SQL usage is parameterized; no obvious SQL injection vector in `limpha/memory.py` and `limpha/episodes.py`.
- ✅ InnerWorld state snapshot uses RW locks when copying shared state (`InnerState` guard in `inner_world/inner_world.go`).

## FORWARD IDEAS (2–3 направления)

1. Add a deterministic "integration harness" that runs a short prompt through C → Go → Python, capturing the full pipeline in one reproducible test. This would immediately detect FFI mismatches and stdout parsing regressions.
2. Build a unified tokenization layer shared across C/Go/Python (single JSON spec + shared loader) and expose a tiny verification tool that asserts vocab size, token mappings, and prompt round-trip integrity.
3. Introduce a small "metrics bus" (protobuf/JSON) for SARTRE + InnerWorld + LIMPHA so that state snapshots flow through a typed contract instead of ad-hoc globals and printf output.

## COMMANDS RUN

- clang --analyze src/ariannabody.c src/arianna_dynamic.c sartre/sartre_kernel.c
- go run -race .   (failed: inner_world build errors)
- cppcheck --enable=all --inconclusive --std=c11 src/ariannabody.c src/arianna_dynamic.c sartre/sartre_kernel.c   (not installed)
- scan-build --status-bugs clang src/ariannabody.c src/arianna_dynamic.c sartre/sartre_kernel.c   (not installed)
