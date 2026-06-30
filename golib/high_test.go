//go:build julia

package main

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

// These tests verify that the REAL Julia high.jl brain matches an INDEPENDENT Go reimplementation
// of the legacy HighMathEngine formulas (high_ref_test.go) — not snapshot constants copied from
// Julia's own output. If high.jl drifted from the legacy semantics, these would fail.

func approxHigh(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func check1(t *testing.T, name, s string, jf func(string) (float64, error), rf func(string) float64) {
	t.Helper()
	got, err := jf(s)
	if err != nil {
		t.Fatalf("%s(%q): %v", name, s, err)
	}
	if want := rf(s); !approxHigh(got, want) {
		t.Errorf("%s(%q): julia=%.12f ref=%.12f", name, s, got, want)
	}
}

func check2(t *testing.T, name, a, b string, jf func(string, string) (float64, error), rf func(string, string) float64) {
	t.Helper()
	got, err := jf(a, b)
	if err != nil {
		t.Fatalf("%s(%q,%q): %v", name, a, b, err)
	}
	if want := rf(a, b); !approxHigh(got, want) {
		t.Errorf("%s(%q,%q): julia=%.12f ref=%.12f", name, a, b, got, want)
	}
}

func checkD(t *testing.T, name string, x float64, jf func(float64) (float64, error), rf func(float64) float64) {
	t.Helper()
	got, err := jf(x)
	if err != nil {
		t.Fatalf("%s(%v): %v", name, x, err)
	}
	if want := rf(x); !approxHigh(got, want) {
		t.Errorf("%s(%v): julia=%.12f ref=%.12f", name, x, got, want)
	}
}

func TestHighJuliaMatchesGoReference(t *testing.T) {
	texts := []string{
		"The Method resonance", "the the the method", "I love this beautiful field",
		"resonance of the sea", "", "a", "the living field, resonance...",
		"люблю красиво отлично здорово", // non-English (RU) emotional weights
		"a a b a a b a a b",              // duplicate n-grams / repeated words
		"hate fear pain die kill betray", // negative + trauma weights
	}
	for _, s := range texts {
		check1(t, "char_entropy", s, HighCharEntropy, refCharEntropy)
		check1(t, "perplexity", s, HighPerplexity, refPerplexity)
		check1(t, "vectorized_entropy", s, HighVectorizedEntropy, func(x string) float64 { e, _ := refVecEntropy(x); return e })
		check1(t, "emotional_score", s, HighEmotionalScore, func(x string) float64 { _, e := refVecEntropy(x); return e })
		check1(t, "valence", s, HighValence, refValence)
		check1(t, "arousal", s, HighArousal, refArousal)
		check1(t, "rhythm_avg", s, HighRhythmAvg, func(x string) float64 { a, _, _ := refRhythm(x); return a })
		check1(t, "rhythm_variance", s, HighRhythmVariance, func(x string) float64 { _, v, _ := refRhythm(x); return v })
		check1(t, "rhythm_pauses", s, HighRhythmPauses, func(x string) float64 { _, _, p := refRhythm(x); return p })
	}

	pairs := [][2]string{
		{"the living field", "the living current"},
		{"resonance of the field", "resonance of the sea"},
		{"I love joy", "beautiful happy hope"},
		{"hate fear pain", "love joy hope"},
		{"a a b a a b", "a a b a a b"},                 // duplicate n-grams (list-length union, not set Jaccard)
		{"люблю жизнь радость", "ненавижу страх боль"}, // non-English emotional alignment
	}
	for _, p := range pairs {
		check2(t, "semantic_distance", p[0], p[1], HighSemanticDistance, refSemDist)
		check2(t, "emotional_alignment", p[0], p[1], HighEmotionalAlignment, refEmoAlign)
		check2(t, "predictive_surprise", p[0], p[1], HighPredictiveSurprise, refPredSurprise)
		got, err := HighNgramOverlap(p[0], p[1], 2)
		if err != nil {
			t.Fatalf("ngram_overlap(%q,%q): %v", p[0], p[1], err)
		}
		if want := refNgramOverlap(p[0], p[1], 2); !approxHigh(got, want) {
			t.Errorf("ngram_overlap(%q,%q): julia=%.12f ref=%.12f", p[0], p[1], got, want)
		}
	}

	got, err := HighResonanceCoupling(0.5, 0.4, 1.2, "I love the calm field", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if want := refResonance(0.5, 0.4, 1.2, "I love the calm field", 1.0); !approxHigh(got, want) {
		t.Errorf("resonance_coupling: julia=%.12f ref=%.12f", got, want)
	}

	for _, x := range []float64{-2, -0.5, 0, 0.5, 2} {
		checkD(t, "sigmoid", x, HighSigmoid, refSigmoid)
		checkD(t, "relu", x, HighReLU, refReLU)
		checkD(t, "tanh", x, HighTanh, refTanh)
	}
}

// TestHighJuliaNULSafe proves strings are passed length-delimited: an embedded NUL is processed,
// not truncated, so Julia sees the same bytes the Go reference does.
func TestHighJuliaNULSafe(t *testing.T) {
	s := "ab\x00cd resonance"
	got, err := HighCharEntropy(s)
	if err != nil {
		t.Fatal(err)
	}
	if want := refCharEntropy(s); !approxHigh(got, want) {
		t.Errorf("NUL truncation: julia=%.12f ref=%.12f (embedded NUL not preserved)", got, want)
	}
}

// TestHighJuliaInit proves the brain boots without error (the error path is real, not a sentinel).
func TestHighJuliaInit(t *testing.T) {
	if err := highStart(); err != nil {
		t.Fatalf("highStart: %v", err)
	}
}

// TestHighJuliaConcurrent proves the dedicated, OS-thread-pinned Julia goroutine serializes
// concurrent callers safely: many goroutines hit the brain at once and every result still
// matches the Go reference (no crash, no race, no wrong value).
func TestHighJuliaConcurrent(t *testing.T) {
	const N = 64
	inputs := []string{"resonance", "the living field", "I love the beautiful sea", "люблю отлично здорово"}
	var wg sync.WaitGroup
	errs := make(chan error, N)
	start := make(chan struct{}) // release all goroutines at once for a true simultaneous hit
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			s := inputs[i%len(inputs)]
			got, err := HighCharEntropy(s)
			if err != nil {
				errs <- err
				return
			}
			if want := refCharEntropy(s); !approxHigh(got, want) {
				errs <- fmt.Errorf("char_entropy(%q): julia=%v ref=%v", s, got, want)
			}
		}(i)
	}
	close(start) // fire all goroutines simultaneously
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
