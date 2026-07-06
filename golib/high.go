// high.go — Arianna's High Mathematical Brain, computed in REAL Julia (libjulia, in-process).
//
// high.jl is a faithful port of the legacy inner_world/high.go HighMathEngine. It carries ALL
// of the engine's analytical metrics — character Shannon entropy, word-level vectorized entropy,
// bigram perplexity, word n-gram overlap, cosine semantic distance, emotional valence/arousal,
// emotional score/alignment, free-energy predictive surprise, Schumann resonance coupling, and
// text rhythm (syllables/variance/pauses) — plus the scalar activations (sigmoid/relu/tanh).
// The legacy softmax/topk are vector sampling helpers (not analytical metrics) and EmotionalDrift
// is a stateful ODE simulator; both belong with the process that drives them, not this analytical
// brain, so they are intentionally out of scope here.
//
// The math runs through libjulia via a thin C shim and is proven faithful to the legacy formulas
// by an independent Go reference (high_ref_test.go). Note the port computes in float64 where the
// legacy engine used float32 — same algorithm, higher precision, so values are algorithm-faithful,
// not bit-identical to legacy.
//
// Robustness: recoverable failures (Julia runtime/eval failure, a missing function, or a Julia
// exception) propagate as a Go error — no magic sentinel is ever consumed as a numeric result.
// (A catastrophic libjulia init can still abort the process; that is libjulia's design, not a
// recoverable path.) Input strings are passed length-delimited (jl_pchar_to_string), so embedded
// NUL bytes are processed, not truncated.
//
// Concurrency: Julia's runtime must not be called concurrently from arbitrary OS threads, so ALL
// libjulia interaction runs on one dedicated goroutine pinned to its OS thread (runtime.LockOSThread).
// The exported Go API is therefore safe to call from any goroutine — every call is marshalled onto
// that single Julia thread and serialized.
//
// Build: libjulia is linked into the DEFAULT trio build — the brain is part of Arianna's body,
// wired into the inner-world processes (overthinking, emotional drift). This makes libjulia a
// hard build/run dependency (a CGO_ENABLED=0 or no-Julia build will not link); `make metabolism`
// derives the Julia prefix from `julia` on PATH, and the #cgo lines below carry a macOS-brew
// default so a bare `go build` / `go test` works on neo.
package main

/*
#cgo CFLAGS: -std=gnu11 -I/opt/homebrew/opt/julia/include/julia -fPIC
#cgo LDFLAGS: -L/opt/homebrew/opt/julia/lib -L/opt/homebrew/opt/julia/lib/julia -Wl,-rpath,/opt/homebrew/opt/julia/lib/julia -Wl,-rpath,/opt/homebrew/opt/julia/lib -ljulia
#include <julia.h>
#include <stdlib.h>
#include <string.h>

static int g_high_ok = 0;

// Boot Julia once and load the high.jl brain into Main. Returns 1 on success, 0 on a recoverable
// fault (an exception while booting or while evaluating high.jl).
static int am_high_setup(const char* src){
    jl_init();
    if (jl_exception_occurred()) return 0;
    jl_eval_string(src);
    if (jl_exception_occurred()) return 0;
    g_high_ok = (jl_main_module != NULL);
    return g_high_ok;
}

// Build a Julia String from a length-delimited byte buffer (NUL-safe; caller must root the slot).
static jl_value_t* am_mkstr(const char* p, int n){
    return (n > 0 && p != NULL) ? jl_pchar_to_string(p, (size_t)n) : jl_cstr_to_string("");
}

// One string argument. *err: 0 ok, 1 brain-not-ready, 2 missing function, 3 julia exception.
static double am_call_s(const char* fn, const char* s, int n, int* err){
    *err = 0;
    if (!g_high_ok) { *err = 1; return 0.0; }
    jl_function_t* f = jl_get_function(jl_main_module, fn);
    if (!f) { *err = 2; return 0.0; }
    jl_value_t* a = NULL;
    JL_GC_PUSH1(&a);              // root before any allocation
    a = am_mkstr(s, n);
    jl_value_t* r = jl_call1(f, a);
    if (r == NULL || jl_exception_occurred()) { *err = 3; JL_GC_POP(); return 0.0; }
    if (!jl_typeis(r, jl_float64_type)) { *err = 4; JL_GC_POP(); return 0.0; }  // defect 1: a non-Float64 return would reinterpret raw bytes as a double
    double out = jl_unbox_float64(r);
    JL_GC_POP();
    return out;
}

// Two string arguments.
static double am_call_ss(const char* fn, const char* a, int an, const char* b, int bn, int* err){
    *err = 0;
    if (!g_high_ok) { *err = 1; return 0.0; }
    jl_function_t* f = jl_get_function(jl_main_module, fn);
    if (!f) { *err = 2; return 0.0; }
    jl_value_t *va = NULL, *vb = NULL;
    JL_GC_PUSH2(&va, &vb);
    va = am_mkstr(a, an);
    vb = am_mkstr(b, bn);
    jl_value_t* r = jl_call2(f, va, vb);
    if (r == NULL || jl_exception_occurred()) { *err = 3; JL_GC_POP(); return 0.0; }
    if (!jl_typeis(r, jl_float64_type)) { *err = 4; JL_GC_POP(); return 0.0; }  // defect 1: a non-Float64 return would reinterpret raw bytes as a double
    double out = jl_unbox_float64(r);
    JL_GC_POP();
    return out;
}

// Two strings + an integer (ngram_overlap).
static double am_call_ssi(const char* fn, const char* a, int an, const char* b, int bn, int k, int* err){
    *err = 0;
    if (!g_high_ok) { *err = 1; return 0.0; }
    jl_function_t* f = jl_get_function(jl_main_module, fn);
    if (!f) { *err = 2; return 0.0; }
    jl_value_t *va = NULL, *vb = NULL, *vk = NULL;
    JL_GC_PUSH3(&va, &vb, &vk);
    va = am_mkstr(a, an);
    vb = am_mkstr(b, bn);
    vk = jl_box_int64(k);
    jl_value_t* args[3] = {va, vb, vk};
    jl_value_t* r = jl_call(f, args, 3);
    if (r == NULL || jl_exception_occurred()) { *err = 3; JL_GC_POP(); return 0.0; }
    if (!jl_typeis(r, jl_float64_type)) { *err = 4; JL_GC_POP(); return 0.0; }  // defect 1: a non-Float64 return would reinterpret raw bytes as a double
    double out = jl_unbox_float64(r);
    JL_GC_POP();
    return out;
}

// One float argument (scalar activations).
static double am_call_d(const char* fn, double x, int* err){
    *err = 0;
    if (!g_high_ok) { *err = 1; return 0.0; }
    jl_function_t* f = jl_get_function(jl_main_module, fn);
    if (!f) { *err = 2; return 0.0; }
    jl_value_t* a = NULL;
    JL_GC_PUSH1(&a);
    a = jl_box_float64(x);
    jl_value_t* r = jl_call1(f, a);
    if (r == NULL || jl_exception_occurred()) { *err = 3; JL_GC_POP(); return 0.0; }
    if (!jl_typeis(r, jl_float64_type)) { *err = 4; JL_GC_POP(); return 0.0; }  // defect 1: a non-Float64 return would reinterpret raw bytes as a double
    double out = jl_unbox_float64(r);
    JL_GC_POP();
    return out;
}

// resonance_coupling(valence, arousal, entropy, external, schumann).
static double am_call_res(double val, double aro, double ent, const char* ext, int extn, double sch, int* err){
    *err = 0;
    if (!g_high_ok) { *err = 1; return 0.0; }
    jl_function_t* f = jl_get_function(jl_main_module, "resonance_coupling");
    if (!f) { *err = 2; return 0.0; }
    jl_value_t *a1=NULL,*a2=NULL,*a3=NULL,*a4=NULL,*a5=NULL;
    JL_GC_PUSH5(&a1,&a2,&a3,&a4,&a5);
    a1 = jl_box_float64(val);
    a2 = jl_box_float64(aro);
    a3 = jl_box_float64(ent);
    a4 = am_mkstr(ext, extn);
    a5 = jl_box_float64(sch);
    jl_value_t* args[5] = {a1,a2,a3,a4,a5};
    jl_value_t* r = jl_call(f, args, 5);
    if (r == NULL || jl_exception_occurred()) { *err = 3; JL_GC_POP(); return 0.0; }
    if (!jl_typeis(r, jl_float64_type)) { *err = 4; JL_GC_POP(); return 0.0; }  // defect 1: a non-Float64 return would reinterpret raw bytes as a double
    double out = jl_unbox_float64(r);
    JL_GC_POP();
    return out;
}
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

//go:embed high.jl
var highJuliaSrc string

// All libjulia calls run on this one OS-thread-pinned goroutine; requests are marshalled to it so
// the exported API is safe from any goroutine and Julia is never touched from two threads at once.
type highJob struct {
	run  func() (float64, error)
	done chan highResult
}

type highResult struct {
	v   float64
	err error
}

var (
	highOnce    sync.Once
	highInitErr error
	highJobs    chan highJob
)

func highStart() error {
	highOnce.Do(func() {
		highJobs = make(chan highJob)
		ready := make(chan error, 1)
		go func() {
			runtime.LockOSThread() // pin Julia to this OS thread for the process's life
			cs := C.CString(highJuliaSrc)
			ok := C.am_high_setup(cs) != 0
			C.free(unsafe.Pointer(cs))
			if !ok {
				ready <- errors.New("high: Julia runtime init or high.jl evaluation failed")
				return
			}
			ready <- nil
			for job := range highJobs {
				v, err := safeHighRun(job.run)
				job.done <- highResult{v, err}
			}
		}()
		highInitErr = <-ready
	})
	return highInitErr
}

// safeHighRun contains a panic in a job so one bad call returns an error to its caller instead of
// crashing the whole organism's process — the long-lived Julia worker must survive a single fault.
func safeHighRun(run func() (float64, error)) (v float64, err error) {
	defer func() {
		if r := recover(); r != nil {
			v, err = 0, fmt.Errorf("high: julia call panicked: %v", r)
		}
	}()
	return run()
}

// highTimeout bounds a single High-brain call (defect 2). A math metric is sub-millisecond; this only
// fires on a pathological/hung Julia call, which libjulia cannot interrupt — so we free the caller (and,
// via the send select, every subsequent caller) instead of deadlocking the organism. done is buffered so
// the worker's late write never blocks.
const highTimeout = 5 * time.Second

// highDo marshals a libjulia closure onto the dedicated Julia thread and waits for the result, bounded
// by highTimeout on both the send and the wait so a stuck call cannot wedge the brain forever.
func highDo(run func() (float64, error)) (float64, error) {
	if err := highStart(); err != nil {
		return 0, err
	}
	done := make(chan highResult, 1)
	timer := time.NewTimer(highTimeout)
	defer timer.Stop()
	select {
	case highJobs <- highJob{run: run, done: done}:
	case <-timer.C:
		return 0, errors.New("high: brain busy — a prior julia call is stuck (send timed out)")
	}
	select {
	case r := <-done:
		return r.v, r.err
	case <-timer.C:
		return 0, errors.New("high: julia call timed out")
	}
}

func highErr(fn string, code C.int) error {
	switch int(code) {
	case 0:
		return nil
	case 1:
		return errors.New("high: brain not initialized")
	case 2:
		return fmt.Errorf("high: julia function %q not found", fn)
	case 4:
		return fmt.Errorf("high: julia %q returned a non-Float64 value", fn)
	default:
		return fmt.Errorf("high: julia %q raised an exception", fn)
	}
}

// highTooLong rejects a string that would overflow C.int (defect 4): a >2GB input truncates or goes
// negative at the C boundary, silently emptying or clipping the text.
func highTooLong(fn string, strs ...string) error {
	for _, s := range strs {
		if len(s) > math.MaxInt32 {
			return fmt.Errorf("high: input to %q too large (%d bytes)", fn, len(s))
		}
	}
	return nil
}

// highBadArg rejects a non-finite float argument (defect 3): don't feed NaN/inf into Julia.
func highBadArg(fn string, xs ...float64) error {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return fmt.Errorf("high: non-finite argument to %q", fn)
		}
	}
	return nil
}

// highResultCheck gates a Julia numeric result on finiteness (defect 3): a NaN/inf from a metric on
// degenerate input is a magic sentinel the file's own contract forbids consuming as a number.
func highResultCheck(fn string, v float64, err error) (float64, error) {
	if err != nil {
		return v, err
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("high: julia %q returned a non-finite result", fn)
	}
	return v, nil
}

func callS(fn, s string) (float64, error) {
	if e := highTooLong(fn, s); e != nil {
		return 0, e
	}
	v, err := highDo(func() (float64, error) {
		cfn := C.CString(fn)
		defer C.free(unsafe.Pointer(cfn))
		cb := C.CBytes([]byte(s))
		defer C.free(cb)
		var cerr C.int
		v := float64(C.am_call_s(cfn, (*C.char)(cb), C.int(len(s)), &cerr))
		return v, highErr(fn, cerr)
	})
	return highResultCheck(fn, v, err)
}

func callSS(fn, a, b string) (float64, error) {
	if e := highTooLong(fn, a, b); e != nil {
		return 0, e
	}
	v, err := highDo(func() (float64, error) {
		cfn := C.CString(fn)
		defer C.free(unsafe.Pointer(cfn))
		ca := C.CBytes([]byte(a))
		defer C.free(ca)
		cbb := C.CBytes([]byte(b))
		defer C.free(cbb)
		var cerr C.int
		v := float64(C.am_call_ss(cfn, (*C.char)(ca), C.int(len(a)), (*C.char)(cbb), C.int(len(b)), &cerr))
		return v, highErr(fn, cerr)
	})
	return highResultCheck(fn, v, err)
}

func callSSI(fn, a, b string, n int) (float64, error) {
	if e := highTooLong(fn, a, b); e != nil {
		return 0, e
	}
	v, err := highDo(func() (float64, error) {
		cfn := C.CString(fn)
		defer C.free(unsafe.Pointer(cfn))
		ca := C.CBytes([]byte(a))
		defer C.free(ca)
		cbb := C.CBytes([]byte(b))
		defer C.free(cbb)
		var cerr C.int
		v := float64(C.am_call_ssi(cfn, (*C.char)(ca), C.int(len(a)), (*C.char)(cbb), C.int(len(b)), C.int(n), &cerr))
		return v, highErr(fn, cerr)
	})
	return highResultCheck(fn, v, err)
}

func callD(fn string, x float64) (float64, error) {
	if e := highBadArg(fn, x); e != nil {
		return 0, e
	}
	v, err := highDo(func() (float64, error) {
		cfn := C.CString(fn)
		defer C.free(unsafe.Pointer(cfn))
		var cerr C.int
		v := float64(C.am_call_d(cfn, C.double(x), &cerr))
		return v, highErr(fn, cerr)
	})
	return highResultCheck(fn, v, err)
}

// ── Analytical metrics (real Julia, faithful to legacy HighMathEngine) ──

// HighCharEntropy — Shannon entropy over characters (legacy CharEntropy).
func HighCharEntropy(s string) (float64, error) { return callS("char_entropy", s) }

// HighVectorizedEntropy — word-level Shannon entropy modulated by emotional intensity (legacy VectorizedEntropy).
func HighVectorizedEntropy(s string) (float64, error) { return callS("vectorized_entropy", s) }

// HighEmotionalScore — mean emotional weight per word (legacy VectorizedEntropy's emotionalScore).
func HighEmotionalScore(s string) (float64, error) { return callS("emotional_score", s) }

// HighPerplexity — bigram perplexity (legacy Perplexity).
func HighPerplexity(s string) (float64, error) { return callS("perplexity", s) }

// HighValence — mean emotional valence over charged words (legacy AnalyzeEmotion valence).
func HighValence(s string) (float64, error) { return callS("analyze_valence", s) }

// HighArousal — emotional density/intensity (legacy AnalyzeEmotion arousal).
func HighArousal(s string) (float64, error) { return callS("analyze_arousal", s) }

// HighRhythmAvg — average syllables per word (legacy TextRhythm avgSyllables).
func HighRhythmAvg(s string) (float64, error) { return callS("rhythm_avg", s) }

// HighRhythmVariance — syllable-count variance / rhythm regularity (legacy TextRhythm variance).
func HighRhythmVariance(s string) (float64, error) { return callS("rhythm_variance", s) }

// HighRhythmPauses — punctuation pause density per word (legacy TextRhythm pauses).
func HighRhythmPauses(s string) (float64, error) { return callS("rhythm_pauses", s) }

// HighSemanticDistance — 1 - cosine over word-count vectors (legacy SemanticDistance).
func HighSemanticDistance(a, b string) (float64, error) { return callSS("semantic_distance", a, b) }

// HighEmotionalAlignment — emotional-score alignment of two texts (legacy EmotionalAlignment).
func HighEmotionalAlignment(a, b string) (float64, error) { return callSS("emotional_alignment", a, b) }

// HighPredictiveSurprise — free-energy proxy: semantic + emotional + entropy error (legacy PredictiveSurprise).
func HighPredictiveSurprise(expected, actual string) (float64, error) {
	return callSS("predictive_surprise", expected, actual)
}

// HighNgramOverlap — n-gram overlap (legacy NgramOverlap): |g1∩g2| / (|g1|+|g2|-overlap), a
// list-length union (faithful to legacy; not a true set Jaccard when n-grams repeat).
func HighNgramOverlap(a, b string, n int) (float64, error) { return callSSI("ngram_overlap", a, b, n) }

// HighResonanceCoupling — Schumann-modulated coupling of an internal (valence, arousal, entropy)
// state with an external text (legacy ResonanceCoupling).
func HighResonanceCoupling(valence, arousal, entropy float64, external string, schumann float64) (float64, error) {
	if e := highBadArg("resonance_coupling", valence, arousal, entropy, schumann); e != nil {
		return 0, e
	}
	if e := highTooLong("resonance_coupling", external); e != nil {
		return 0, e
	}
	v, err := highDo(func() (float64, error) {
		cb := C.CBytes([]byte(external))
		defer C.free(cb)
		var cerr C.int
		v := float64(C.am_call_res(C.double(valence), C.double(arousal), C.double(entropy),
			(*C.char)(cb), C.int(len(external)), C.double(schumann), &cerr))
		return v, highErr("resonance_coupling", cerr)
	})
	return highResultCheck("resonance_coupling", v, err)
}

// ── Scalar activations (legacy Sigmoid/Tanh/ReLU) ──

// HighSigmoid — logistic activation (legacy Sigmoid).
func HighSigmoid(x float64) (float64, error) { return callD("sigmoid", x) }

// HighReLU — rectified linear activation (legacy ReLU).
func HighReLU(x float64) (float64, error) { return callD("relu", x) }

// HighTanh — hyperbolic-tangent activation (legacy Tanh).
func HighTanh(x float64) (float64, error) { return callD("tanh_act", x) }
