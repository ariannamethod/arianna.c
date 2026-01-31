package main

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

func TestTongueE2E(t *testing.T) {
	weightsPath := os.Getenv("TONGUE_WEIGHTS")
	if weightsPath == "" {
		weightsPath = "weights/arianna_1b_step3000_q4_0.gguf"
	}

	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skipf("weights not found at %s (set TONGUE_WEIGHTS env)", weightsPath)
	}

	// Load GGUF
	fmt.Printf("[test] loading GGUF from %s\n", weightsPath)
	gguf, err := LoadGGUF(weightsPath)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	fmt.Printf("[test] GGUF: %d tensors loaded\n", len(gguf.Tensors))

	// Load model
	model, err := LoadLlamaModel(gguf)
	if err != nil {
		t.Fatalf("LoadLlamaModel: %v", err)
	}
	fmt.Printf("[test] Model: %d layers, %d dim, %d heads, %d kv_heads, %d vocab, seq_len=%d\n",
		model.Config.NumLayers, model.Config.EmbedDim,
		model.Config.NumHeads, model.Config.NumKVHeads,
		model.Config.VocabSize, model.Config.SeqLen)

	// Load tokenizer
	tokenizer := NewTokenizer(&gguf.Meta)
	fmt.Printf("[test] Tokenizer: vocab=%d bos=%d eos=%d\n",
		tokenizer.VocabSize, tokenizer.BosID, tokenizer.EosID)

	// Encode prompt
	prompt := "who are you?"
	tokens := tokenizer.Encode(prompt, false)
	fmt.Printf("[test] Prompt: %q → %d tokens: %v\n", prompt, len(tokens), tokens)

	if len(tokens) == 0 {
		t.Fatal("tokenizer returned 0 tokens")
	}

	// Build sequence: BOS + prompt
	var allTokens []int
	if tokenizer.BosID >= 0 {
		allTokens = append(allTokens, tokenizer.BosID)
	}
	allTokens = append(allTokens, tokens...)

	// Feed prompt
	model.Reset()
	pos := 0
	fmt.Printf("[test] Feeding %d prompt tokens...\n", len(allTokens))
	t0 := time.Now()
	for _, tok := range allTokens {
		model.Forward(tok, pos)
		pos++
	}
	promptTime := time.Since(t0)
	fmt.Printf("[test] Prompt: %v (%.1f tok/s)\n",
		promptTime, float64(len(allTokens))/promptTime.Seconds())

	// Generate
	rng := rand.New(rand.NewSource(42))
	_ = rng
	maxTokens := 100
	temp := float32(0.9)

	fmt.Printf("[test] Generating (temp=%.1f, max=%d)...\n", temp, maxTokens)
	fmt.Print("[tongue]: ")
	t1 := time.Now()
	genCount := 0
	var output string

	for i := 0; i < maxTokens; i++ {
		next := sampleTopK(model.State.Logits, model.Config.VocabSize, temp, 50)

		if next == tokenizer.EosID {
			break
		}

		piece := tokenizer.DecodeToken(next)
		fmt.Print(piece)
		output += piece

		model.Forward(next, pos)
		pos++
		genCount++
	}

	genTime := time.Since(t1)
	fmt.Printf("\n[test] Generated %d tokens in %v (%.2f tok/s)\n",
		genCount, genTime, float64(genCount)/genTime.Seconds())
	fmt.Printf("[test] Output: %q\n", output)

	if genCount == 0 {
		t.Fatal("generated 0 tokens")
	}
}
