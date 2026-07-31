package main

import (
	"math/rand"
	"testing"
)

func TestSampleTopKBoundsUserRequestedK(t *testing.T) {
	engine := &Engine{
		model: &LlamaModel{
			Config: LlamaConfig{VocabSize: 3},
			State:  LlamaState{Logits: []float32{0.1, 0.9, 0.2}},
		},
		rng: rand.New(rand.NewSource(1)),
	}

	got := engine.sampleTopK(0.8, 1<<30)
	if got < 0 || got >= engine.model.Config.VocabSize {
		t.Fatalf("sampleTopK returned token %d outside vocab", got)
	}
}
