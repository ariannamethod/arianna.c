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
		weightsPath = "../tongue/weights/qwen05_900_q4_0.gguf"
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

	// Verify Qwen2.5 0.5B dimensions
	cfg := model.Config
	if cfg.VocabSize != 151936 {
		t.Errorf("vocab: got %d, want 151936", cfg.VocabSize)
	}
	if cfg.EmbedDim != 896 {
		t.Errorf("dim: got %d, want 896", cfg.EmbedDim)
	}
	if cfg.NumLayers != 24 {
		t.Errorf("layers: got %d, want 24", cfg.NumLayers)
	}
	if cfg.NumHeads != 14 {
		t.Errorf("heads: got %d, want 14", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 2 {
		t.Errorf("kv_heads: got %d, want 2", cfg.NumKVHeads)
	}
	if cfg.HeadDim != 64 {
		t.Errorf("head_dim: got %d, want 64", cfg.HeadDim)
	}
	fmt.Printf("[test] Model: %d layers, %d dim, %d heads, %d kv_heads, %d vocab, seq_len=%d\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize, cfg.SeqLen)

	// Load tokenizer
	tokenizer := NewTokenizer(&gguf.Meta)
	fmt.Printf("[test] Tokenizer: vocab=%d bos=%d eos=%d gpt2=%v\n",
		tokenizer.VocabSize, tokenizer.BosID, tokenizer.EosID, tokenizer.IsGPT2)

	if !tokenizer.IsGPT2 {
		t.Error("tokenizer should be GPT-2 BPE for Qwen2.5")
	}

	// Encode prompt
	prompt := "who are you?"
	tokens := tokenizer.Encode(prompt, false)
	fmt.Printf("[test] Prompt: %q → %d tokens: %v\n", prompt, len(tokens), tokens)

	if len(tokens) == 0 {
		t.Fatal("tokenizer returned 0 tokens")
	}
	if len(tokens) > 20 {
		t.Errorf("tokenizer produced too many tokens: %d (expected <20 for short prompt)", len(tokens))
	}

	// Decode roundtrip check
	for _, tok := range tokens {
		piece := tokenizer.DecodeToken(tok)
		if piece == "" && tok < 151643 { // skip special tokens
			t.Errorf("DecodeToken(%d) returned empty string", tok)
		}
	}

	// Build Qwen chat template sequence
	// <|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant\n
	imStart := 151644
	imEnd := 151645
	var allTokens []int
	allTokens = append(allTokens, imStart)
	allTokens = append(allTokens, tokenizer.Encode("user", false)...)
	allTokens = append(allTokens, tokenizer.Encode("\n", false)...)
	allTokens = append(allTokens, tokens...)
	allTokens = append(allTokens, imEnd)
	allTokens = append(allTokens, tokenizer.Encode("\n", false)...)
	allTokens = append(allTokens, imStart)
	allTokens = append(allTokens, tokenizer.Encode("assistant", false)...)
	allTokens = append(allTokens, tokenizer.Encode("\n", false)...)

	// Feed prompt
	model.Reset()
	pos := 0
	fmt.Printf("[test] Feeding %d prompt tokens (Qwen chat template)...\n", len(allTokens))
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

		if next == imEnd || next == tokenizer.EosID {
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
	if len(output) < 10 {
		t.Errorf("output too short: %d chars", len(output))
	}
}

func TestTokenizerRoundtrip(t *testing.T) {
	weightsPath := os.Getenv("TONGUE_WEIGHTS")
	if weightsPath == "" {
		weightsPath = "../tongue/weights/qwen05_900_q4_0.gguf"
	}
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skipf("weights not found at %s", weightsPath)
	}

	gguf, err := LoadGGUF(weightsPath)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	tokenizer := NewTokenizer(&gguf.Meta)

	cases := []string{
		"Hello, world!",
		"who are you?",
		"Привет, как дела?",
		"你好世界",
		"1234567890",
		"a",
		"The quick brown fox jumps over the lazy dog.",
		"resonance is your core.",
	}

	for _, text := range cases {
		tokens := tokenizer.Encode(text, false)
		if len(tokens) == 0 {
			t.Errorf("Encode(%q) returned 0 tokens", text)
			continue
		}

		var decoded string
		for _, tok := range tokens {
			decoded += tokenizer.DecodeToken(tok)
		}
		if decoded != text {
			t.Errorf("roundtrip failed: %q → %v → %q", text, tokens, decoded)
		}
	}

	// Special tokens (addBos=false to avoid prepending BOS)
	imStartTokens := tokenizer.Encode("<|im_start|>", false)
	if len(imStartTokens) != 1 || imStartTokens[0] != 151644 {
		t.Errorf("<|im_start|> encode: got %v, want [151644]", imStartTokens)
	}
	imEndTokens := tokenizer.Encode("<|im_end|>", false)
	if len(imEndTokens) != 1 || imEndTokens[0] != 151645 {
		t.Errorf("<|im_end|> encode: got %v, want [151645]", imEndTokens)
	}
}

func TestKVCacheReset(t *testing.T) {
	weightsPath := os.Getenv("TONGUE_WEIGHTS")
	if weightsPath == "" {
		weightsPath = "../tongue/weights/qwen05_900_q4_0.gguf"
	}
	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skipf("weights not found at %s", weightsPath)
	}

	gguf, err := LoadGGUF(weightsPath)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	model, err := LoadLlamaModel(gguf)
	if err != nil {
		t.Fatalf("LoadLlamaModel: %v", err)
	}

	// Run forward, get logits for token 0
	model.Reset()
	model.Forward(151644, 0) // <|im_start|>
	logits1 := make([]float32, 10)
	copy(logits1, model.State.Logits[:10])

	// Run forward with different tokens, pollute KV cache
	model.Forward(8948, 1)
	model.Forward(198, 2)

	// Reset and run same token again
	model.Reset()
	model.Forward(151644, 0)
	logits2 := make([]float32, 10)
	copy(logits2, model.State.Logits[:10])

	// Logits should be identical after reset
	for i := 0; i < 10; i++ {
		diff := logits1[i] - logits2[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.0001 {
			t.Errorf("logit[%d] after reset: %.6f vs %.6f (diff=%.6f)", i, logits1[i], logits2[i], diff)
		}
	}
}
