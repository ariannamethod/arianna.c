package main

// tongue.go — CGO bridge: Arianna's Tongue (1.1B TinyLlama GGUF)
//
// This is the voice. The only interface with the world.
// 1.1B parameters fine-tuned on Arianna's identity, philosophy, and voice.
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

	var err error
	gGGUF, err = LoadGGUF(path)
	if err != nil {
		fmt.Printf("[tongue] ERROR loading GGUF: %v\n", err)
		return -1
	}

	gModel, err = LoadLlamaModel(gGGUF)
	if err != nil {
		fmt.Printf("[tongue] ERROR loading model: %v\n", err)
		return -1
	}

	gTokenizer = NewTokenizer(&gGGUF.Meta)

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

	// Build token sequence: BOS + anchor + user input
	var allTokens []int

	// 1. BOS
	if gTokenizer.BosID >= 0 {
		allTokens = append(allTokens, gTokenizer.BosID)
	}

	// 2. Anchor prompt (identity + metabolism + heuristics)
	if anchorPrompt != "" {
		anchorTokens := gTokenizer.Encode(anchorPrompt, false)
		allTokens = append(allTokens, anchorTokens...)
	}

	// 3. User prompt
	userTokens := gTokenizer.Encode(prompt, false)
	allTokens = append(allTokens, userTokens...)

	// Reset KV cache for new generation
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
	temp := float32(temperature) * gTempMod
	if temp < tempFloor {
		temp = tempFloor
	}
	tp := float32(topP)

	var output []byte
	genCount := 0
	maxTok := int(maxTokens)
	maxOut := int(maxOutputLen) - 1

	for i := 0; i < maxTok && len(output) < maxOut; i++ {
		// Apply logit scale
		if gLogitScale != 1.0 {
			for j := 0; j < gModel.Config.VocabSize; j++ {
				gModel.State.Logits[j] *= gLogitScale
			}
		}

		// Apply exploratory bias
		if gExploresBias != 0 {
			for j := 0; j < gModel.Config.VocabSize; j++ {
				noise := float32(math.Sin(float64(j)*0.12345 + float64(pos)*0.54321))
				gModel.State.Logits[j] += gExploresBias * noise
			}
		}

		// Sample
		var next int
		if tp < 1.0 {
			next = sampleTopP(gModel.State.Logits, gModel.Config.VocabSize, temp, tp)
		} else {
			next = sampleTopK(gModel.State.Logits, gModel.Config.VocabSize, temp, 50)
		}

		// Check for EOS
		if next == gTokenizer.EosID {
			break
		}

		// Decode token
		piece := gTokenizer.DecodeToken(next)
		output = append(output, []byte(piece)...)

		// Forward next token
		gModel.Forward(next, pos)
		pos++
		genCount++

		if pos >= gModel.Config.SeqLen {
			break
		}
	}

	// Copy to C buffer
	if len(output) > maxOut {
		output = output[:maxOut]
	}
	if len(output) > 0 {
		cOutput := (*[1 << 30]byte)(unsafe.Pointer(outputC))[:len(output)+1:len(output)+1]
		copy(cOutput, output)
		cOutput[len(output)] = 0
	} else {
		cOutput := (*[1]byte)(unsafe.Pointer(outputC))
		cOutput[0] = 0
	}

	return C.int(genCount)
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

	// Sort indices by logit value descending
	type idxVal struct {
		idx int
		val float32
	}
	sorted := make([]idxVal, vocab)
	for i := 0; i < vocab; i++ {
		sorted[i] = idxVal{i, logits[i]}
	}

	// Partial sort: we only need enough for top-p mass
	// Use simple selection for now (good enough for ~32k vocab)
	for i := 0; i < vocab-1; i++ {
		best := i
		for j := i + 1; j < vocab; j++ {
			if sorted[j].val > sorted[best].val {
				best = j
			}
		}
		sorted[i], sorted[best] = sorted[best], sorted[i]

		// Early exit if we've accumulated enough probability mass
		// (heuristic: top 256 is usually enough for p=0.95)
		if i > 256 {
			break
		}
	}

	// Softmax and find nucleus
	maxVal := sorted[0].val
	probs := make([]float32, vocab)
	var sum float32
	for i := 0; i < vocab; i++ {
		probs[i] = float32(math.Exp(float64((sorted[i].val - maxVal) / temp)))
		sum += probs[i]
	}
	for i := 0; i < vocab; i++ {
		probs[i] /= sum
	}

	// Find nucleus size
	var cumsum float32
	nucleusSize := vocab
	for i := 0; i < vocab; i++ {
		cumsum += probs[i]
		if cumsum >= topP {
			nucleusSize = i + 1
			break
		}
	}

	// Renormalize and sample
	sum = 0
	for i := 0; i < nucleusSize; i++ {
		sum += probs[i]
	}

	r := gRNG.Float32() * sum
	var cdf float32
	for i := 0; i < nucleusSize; i++ {
		cdf += probs[i]
		if r <= cdf {
			return sorted[i].idx
		}
	}
	return sorted[0].idx
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
