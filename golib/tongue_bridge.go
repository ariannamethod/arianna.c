package main

// tongue.go — CGO bridge: Arianna's Tongue (Qwen2.5 0.5B GGUF)
//
// This is the voice. The only interface with the world.
// 0.5B parameters fine-tuned on Arianna's identity, philosophy, and voice.
// 29 languages. 336MB. Poet.
//
// Build as shared library:
//   go build -buildmode=c-shared -o libtongue.so .
//
// C interface matches D12Bridge pattern for drop-in replacement.
//
// This is not inference. This is breathing.

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
	"unsafe"
)

// Global state (singleton — one Tongue per organism)
var (
	gModel     *LlamaModel
	gTokenizer *Tokenizer
	gGGUF      *GGUFFile
	gMu        sync.Mutex
	gRNG       *rand.Rand

	// Modulation from Arianna ecosystem
	gTempMod       float32 = 1.0
	gLogitScale    float32 = 1.0
	gExploresBias  float32 = 0.0

	// Temperature floor: Tongue never freezes
	tempFloor float32 = 0.9

	// Repetition penalty: prevents loops and language drift
	repPenalty float32 = 1.15  // >1.0 penalizes repetition
	repWindow  int     = 64    // look-back window for recent tokens
)

func init() {
	gRNG = rand.New(rand.NewSource(time.Now().UnixNano()))
}

// ============================================================
// Lifecycle
// ============================================================

//export tongue_init
func tongue_init(weightsPath *C.char) C.int {
	gMu.Lock()
	defer gMu.Unlock()

	path := C.GoString(weightsPath)
	fmt.Printf("[tongue] loading GGUF from %s\n", path)

	// Heavy work runs in a goroutine (full Go stack) to avoid
	// cgo callback stack overflow when loading 607MB+ GGUF.
	type initResult struct {
		gguf      *GGUFFile
		model     *LlamaModel
		tokenizer *Tokenizer
		err       error
	}
	ch := make(chan initResult, 1)
	go func() {
		var r initResult
		r.gguf, r.err = LoadGGUF(path)
		if r.err != nil {
			ch <- r
			return
		}
		r.model, r.err = LoadLlamaModel(r.gguf)
		if r.err != nil {
			ch <- r
			return
		}
		r.tokenizer = NewTokenizer(&r.gguf.Meta)
		ch <- r
	}()
	r := <-ch

	if r.err != nil {
		fmt.Printf("[tongue] ERROR: %v\n", r.err)
		return -1
	}

	gGGUF = r.gguf
	gModel = r.model
	gTokenizer = r.tokenizer

	fmt.Printf("[tongue] initialized: %d layers, %d dim, %d vocab, temp_floor=%.1f\n",
		gModel.Config.NumLayers, gModel.Config.EmbedDim,
		gModel.Config.VocabSize, tempFloor)

	return 0
}

//export tongue_free
func tongue_free() {
	gMu.Lock()
	defer gMu.Unlock()

	gModel = nil
	gTokenizer = nil
	gGGUF = nil
	fmt.Println("[tongue] freed")
}

// ============================================================
// Modulation — signals from Arianna ecosystem
// ============================================================

//export tongue_set_temperature_mod
func tongue_set_temperature_mod(mod C.float) {
	gTempMod = float32(mod)
}

//export tongue_set_logit_scale
func tongue_set_logit_scale(scale C.float) {
	gLogitScale = float32(scale)
}

//export tongue_set_exploratory_bias
func tongue_set_exploratory_bias(bias C.float) {
	gExploresBias = float32(bias)
}

//export tongue_set_temp_floor
func tongue_set_temp_floor(floor C.float) {
	tempFloor = float32(floor)
}

//export tongue_set_rep_penalty
func tongue_set_rep_penalty(penalty C.float, window C.int) {
	repPenalty = float32(penalty)
	repWindow = int(window)
}

// ============================================================
// Generation
// ============================================================

//export tongue_reset
func tongue_reset() {
	gMu.Lock()
	defer gMu.Unlock()
	if gModel != nil {
		gModel.Reset()
	}
}

//export tongue_generate
func tongue_generate(
	promptC *C.char,
	outputC *C.char, maxOutputLen C.int,
	maxTokens C.int,
	temperature C.float, topP C.float,
	anchorPromptC *C.char,
) C.int {
	gMu.Lock()
	defer gMu.Unlock()

	if gModel == nil || gTokenizer == nil {
		return 0
	}

	prompt := C.GoString(promptC)
	anchorPrompt := ""
	if anchorPromptC != nil {
		anchorPrompt = C.GoString(anchorPromptC)
	}

	maxTok := int(maxTokens)
	maxOut := int(maxOutputLen) - 1
	temp := float32(temperature) * gTempMod
	if temp < tempFloor {
		temp = tempFloor
	}
	tp := float32(topP)

	// Run generation in goroutine (full Go stack) to avoid cgo stack limits
	type genResult struct {
		output   []byte
		genCount int
	}
	ch := make(chan genResult, 1)
	go func() {
		// Build token sequence: BOS + anchor + user input
		var allTokens []int

		if gTokenizer.BosID >= 0 {
			allTokens = append(allTokens, gTokenizer.BosID)
		}
		if anchorPrompt != "" {
			anchorTokens := gTokenizer.Encode(anchorPrompt, false)
			allTokens = append(allTokens, anchorTokens...)
		}
		userTokens := gTokenizer.Encode(prompt, false)
		allTokens = append(allTokens, userTokens...)

		gModel.Reset()

		// Feed all tokens through transformer
		pos := 0
		for _, tok := range allTokens {
			gModel.Forward(tok, pos)
			pos++
			if pos >= gModel.Config.SeqLen-1 {
				break
			}
		}

		// Generate
		var output []byte
		genCount := 0
		graceLimit := 32
		inGrace := false
		recentTokens := make([]int, 0, repWindow)

		for i := 0; i < maxTok+graceLimit && len(output) < maxOut; i++ {
			if i >= maxTok && !inGrace {
				inGrace = true
			}
			if inGrace {
				if len(output) > 0 {
					last := output[len(output)-1]
					if last == '.' || last == '!' || last == '?' || last == '\n' {
						break
					}
				}
			}

			if gLogitScale != 1.0 {
				for j := 0; j < gModel.Config.VocabSize; j++ {
					gModel.State.Logits[j] *= gLogitScale
				}
			}

			if repPenalty > 1.0 && len(recentTokens) > 0 {
				for _, tok := range recentTokens {
					if tok >= 0 && tok < gModel.Config.VocabSize {
						logit := gModel.State.Logits[tok]
						if logit > 0 {
							gModel.State.Logits[tok] = logit / repPenalty
						} else {
							gModel.State.Logits[tok] = logit * repPenalty
						}
					}
				}
			}

			if gExploresBias != 0 {
				for j := 0; j < gModel.Config.VocabSize; j++ {
					noise := float32(math.Sin(float64(j)*0.12345 + float64(pos)*0.54321))
					gModel.State.Logits[j] += gExploresBias * noise
				}
			}

			var next int
			if tp < 1.0 {
				next = sampleTopP(gModel.State.Logits, gModel.Config.VocabSize, temp, tp)
			} else {
				next = sampleTopK(gModel.State.Logits, gModel.Config.VocabSize, temp, 50)
			}

			recentTokens = append(recentTokens, next)
			if len(recentTokens) > repWindow {
				recentTokens = recentTokens[1:]
			}

			if next == gTokenizer.EosID {
				break
			}

			piece := gTokenizer.DecodeToken(next)
			output = append(output, []byte(piece)...)

			gModel.Forward(next, pos)
			pos++
			genCount++

			if pos >= gModel.Config.SeqLen {
				break
			}
		}
		ch <- genResult{output, genCount}
	}()
	r := <-ch

	// Copy to C buffer
	if len(r.output) > maxOut {
		r.output = r.output[:maxOut]
	}
	if len(r.output) > 0 {
		cOutput := (*[1 << 30]byte)(unsafe.Pointer(outputC))[:len(r.output)+1:len(r.output)+1]
		copy(cOutput, r.output)
		cOutput[len(r.output)] = 0
	} else {
		cOutput := (*[1]byte)(unsafe.Pointer(outputC))
		cOutput[0] = 0
	}

	return C.int(r.genCount)
}

// ============================================================
// Tokenization
// ============================================================

//export tongue_encode
func tongue_encode(textC *C.char, idsOut *C.int, maxTokens C.int) C.int {
	if gTokenizer == nil {
		return 0
	}
	text := C.GoString(textC)
	ids := gTokenizer.Encode(text, false)

	max := int(maxTokens)
	if len(ids) > max {
		ids = ids[:max]
	}

	// Copy to C array
	out := (*[1 << 20]C.int)(unsafe.Pointer(idsOut))[:len(ids):len(ids)]
	for i, id := range ids {
		out[i] = C.int(id)
	}
	return C.int(len(ids))
}

//export tongue_decode_token
func tongue_decode_token(id C.int) *C.char {
	if gTokenizer == nil {
		return C.CString("")
	}
	piece := gTokenizer.DecodeToken(int(id))
	return C.CString(piece)
}

// ============================================================
// State queries
// ============================================================

//export tongue_get_vocab_size
func tongue_get_vocab_size() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.VocabSize)
}

//export tongue_get_dim
func tongue_get_dim() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.EmbedDim)
}

//export tongue_get_seq_len
func tongue_get_seq_len() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.SeqLen)
}

//export tongue_get_num_layers
func tongue_get_num_layers() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.NumLayers)
}

// tongue_get_logits_into copies logits into caller-provided C buffer.
// Returns number of floats written.
//
//export tongue_get_logits_into
func tongue_get_logits_into(out *C.float, maxLen C.int) C.int {
	if gModel == nil || out == nil {
		return 0
	}
	n := gModel.Config.VocabSize
	if int(maxLen) < n {
		n = int(maxLen)
	}
	cSlice := (*[1 << 24]C.float)(unsafe.Pointer(out))[:n:n]
	for i := 0; i < n; i++ {
		cSlice[i] = C.float(gModel.State.Logits[i])
	}
	return C.int(n)
}

// tongue_get_hidden_into copies hidden state into caller-provided C buffer.
// Returns number of floats written.
//
//export tongue_get_hidden_into
func tongue_get_hidden_into(out *C.float, maxLen C.int) C.int {
	if gModel == nil || out == nil {
		return 0
	}
	n := gModel.Config.EmbedDim
	if int(maxLen) < n {
		n = int(maxLen)
	}
	cSlice := (*[1 << 24]C.float)(unsafe.Pointer(out))[:n:n]
	for i := 0; i < n; i++ {
		cSlice[i] = C.float(gModel.State.X[i])
	}
	return C.int(n)
}

// ============================================================
// Sampling implementations
// ============================================================

func sampleTopK(logits []float32, vocab int, temp float32, topK int) int {
	if temp <= 0 {
		return argmax(logits, vocab)
	}
	if topK > vocab {
		topK = vocab
	}

	// Find top-k indices
	type idxVal struct {
		idx int
		val float32
	}
	top := make([]idxVal, topK)
	for i := 0; i < topK; i++ {
		top[i] = idxVal{-1, -1e30}
	}

	for i := 0; i < vocab; i++ {
		if logits[i] > top[topK-1].val {
			top[topK-1] = idxVal{i, logits[i]}
			// Bubble up
			for j := topK - 1; j > 0 && top[j].val > top[j-1].val; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}

	// Softmax over top-k
	maxVal := top[0].val
	probs := make([]float32, topK)
	var sum float32
	for i := 0; i < topK; i++ {
		if top[i].idx < 0 {
			break
		}
		probs[i] = float32(math.Exp(float64((top[i].val - maxVal) / temp)))
		sum += probs[i]
	}

	// Sample
	r := gRNG.Float32() * sum
	var cdf float32
	for i := 0; i < topK; i++ {
		cdf += probs[i]
		if r <= cdf {
			return top[i].idx
		}
	}
	return top[0].idx
}

func sampleTopP(logits []float32, vocab int, temp float32, topP float32) int {
	if temp <= 0 {
		return argmax(logits, vocab)
	}

	// Apply temperature and compute softmax in one pass
	maxVal := logits[0]
	for i := 1; i < vocab; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	type idxProb struct {
		idx  int
		prob float32
	}
	candidates := make([]idxProb, vocab)
	var sum float32
	for i := 0; i < vocab; i++ {
		p := float32(math.Exp(float64((logits[i] - maxVal) / temp)))
		candidates[i] = idxProb{i, p}
		sum += p
	}

	// Normalize
	invSum := float32(1.0) / sum
	for i := range candidates {
		candidates[i].prob *= invSum
	}

	// Sort by probability descending using stdlib (O(n log n) vs O(n*256))
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	// Find nucleus and sample
	var cumsum float32
	for i := range candidates {
		cumsum += candidates[i].prob
		if cumsum >= topP {
			// Renormalize nucleus and sample
			r := gRNG.Float32() * cumsum
			var cdf float32
			for j := 0; j <= i; j++ {
				cdf += candidates[j].prob
				if r <= cdf {
					return candidates[j].idx
				}
			}
			return candidates[0].idx
		}
	}
	return candidates[0].idx
}

func argmax(logits []float32, n int) int {
	best := 0
	for i := 1; i < n; i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// ============================================================
// Required for c-shared build
// ============================================================

func main() {}
