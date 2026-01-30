package main

// tokenizer.go — SentencePiece BPE tokenizer from GGUF metadata
//
// TinyLlama uses SentencePiece BPE with 32000 vocab.
// The token list, scores, and types are stored in GGUF metadata.
//
// Token types:
//   1 = normal
//   2 = unknown (<unk>)
//   3 = control (<s>, </s>)
//   6 = byte fallback (<0x00>...<0xFF>)
//
// Encoding: greedy BPE merge. We use the standard approach:
//   1. Split text into UTF-8 bytes
//   2. Map bytes to byte tokens ▁ prefixed (SentencePiece adds ▁ for space)
//   3. Iteratively merge highest-score adjacent pairs

import (
	"fmt"
	"sort"
	"strings"
)

// Tokenizer handles SentencePiece BPE encoding/decoding
type Tokenizer struct {
	Vocab          []string
	Scores         []float32
	Types          []int32
	VocabSize      int
	BosID          int
	EosID          int
	AddSpacePrefix bool

	// Lookup table for encoding
	tokenToID map[string]int
	// Byte fallback tokens
	byteTokens [256]int
}

// NewTokenizer creates a tokenizer from GGUF metadata
func NewTokenizer(meta *GGUFMetadata) *Tokenizer {
	t := &Tokenizer{
		Vocab:          meta.TokenList,
		Scores:         meta.TokenScores,
		Types:          meta.TokenTypes,
		VocabSize:      meta.VocabSize,
		BosID:          meta.BosID,
		EosID:          meta.EosID,
		AddSpacePrefix: meta.AddSpacePrefix,
	}

	// Build lookup table
	t.tokenToID = make(map[string]int, t.VocabSize)
	for i, tok := range t.Vocab {
		t.tokenToID[tok] = i
	}

	// Map byte fallback tokens
	for i := 0; i < 256; i++ {
		name := fmt.Sprintf("<0x%02X>", i)
		if id, ok := t.tokenToID[name]; ok {
			t.byteTokens[i] = id
		} else {
			t.byteTokens[i] = -1
		}
	}

	fmt.Printf("[tongue/tokenizer] vocab=%d bos=%d eos=%d add_space_prefix=%v\n", t.VocabSize, t.BosID, t.EosID, t.AddSpacePrefix)
	return t
}

// Encode converts text to token IDs using BPE
func (t *Tokenizer) Encode(text string, addBos bool) []int {
	var tokens []int

	if addBos && t.BosID >= 0 {
		tokens = append(tokens, t.BosID)
	}

	if len(text) == 0 {
		return tokens
	}

	// SentencePiece: prepend space to input only if add_space_prefix is set
	if t.AddSpacePrefix && text[0] != ' ' {
		text = " " + text
	}

	// Start with each character as a token (using SentencePiece convention)
	// SentencePiece replaces spaces with ▁ (U+2581)
	text = strings.ReplaceAll(text, " ", "▁")

	// Initial tokenization: try to match each character/codepoint
	var chars []int
	for _, b := range []byte(text) {
		// Try single byte as part of UTF-8, use byte fallback
		if t.byteTokens[b] >= 0 {
			chars = append(chars, t.byteTokens[b])
		}
	}

	// Actually, let's do proper BPE encoding.
	// Start with the UTF-8 text split into initial tokens.
	// Try to find each single character in vocab first.
	symbols := t.initialTokenize(text)

	// BPE merge loop
	for {
		bestScore := float32(-1e30)
		bestIdx := -1

		// Find best adjacent pair to merge
		for i := 0; i < len(symbols)-1; i++ {
			merged := symbols[i] + symbols[i+1]
			if id, ok := t.tokenToID[merged]; ok {
				score := t.Scores[id]
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break // No more merges possible
		}

		// Merge the best pair
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Convert symbols to token IDs
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym]; ok {
			tokens = append(tokens, id)
		} else {
			// Fall back to byte tokens
			for _, b := range []byte(sym) {
				if t.byteTokens[b] >= 0 {
					tokens = append(tokens, t.byteTokens[b])
				}
			}
		}
	}

	return tokens
}

// initialTokenize splits text into initial symbols for BPE
func (t *Tokenizer) initialTokenize(text string) []string {
	// Try to split into individual characters that exist in vocab
	// For SentencePiece, the ▁ character is part of the token
	var symbols []string

	runes := []rune(text)
	i := 0
	for i < len(runes) {
		// Try longest match first (up to some reasonable length)
		matched := false
		maxLen := len(runes) - i
		if maxLen > 32 {
			maxLen = 32
		}
		// Start with single characters
		ch := string(runes[i])
		if _, ok := t.tokenToID[ch]; ok {
			symbols = append(symbols, ch)
			i++
			matched = true
		}

		if !matched {
			// Fall back to byte representation
			for _, b := range []byte(string(runes[i])) {
				byteStr := fmt.Sprintf("<0x%02X>", b)
				symbols = append(symbols, byteStr)
			}
			i++
		}
	}

	return symbols
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= t.VocabSize {
			continue
		}
		piece := t.Vocab[id]

		// Skip control tokens
		if t.Types != nil && id < len(t.Types) && t.Types[id] == 3 {
			continue
		}

		// Handle byte fallback tokens
		if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
			var b byte
			fmt.Sscanf(piece, "<0x%02X>", &b)
			sb.WriteByte(b)
			continue
		}

		// SentencePiece: ▁ -> space
		piece = strings.ReplaceAll(piece, "▁", " ")
		sb.WriteString(piece)
	}

	result := sb.String()
	// Trim leading space that was added during encoding (only if add_space_prefix was used)
	if t.AddSpacePrefix && len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}
	return result
}

// DecodeToken decodes a single token ID
func (t *Tokenizer) DecodeToken(id int) string {
	if id < 0 || id >= t.VocabSize {
		return ""
	}
	piece := t.Vocab[id]

	// Handle byte fallback
	if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
		var b byte
		fmt.Sscanf(piece, "<0x%02X>", &b)
		return string([]byte{b})
	}

	// SentencePiece: ▁ -> space
	piece = strings.ReplaceAll(piece, "▁", " ")
	return piece
}

// ChatTemplate IDs for TinyLlama chat format
// TinyLlama uses: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n
func (t *Tokenizer) FindSpecialToken(name string) int {
	// Check GGUF chat template tokens
	variants := []string{
		name,
		"<|" + name + "|>",
		"<" + name + ">",
	}
	for _, v := range variants {
		if id, ok := t.tokenToID[v]; ok {
			return id
		}
	}
	return -1
}

// DebugTokenize shows tokens for debugging
func (t *Tokenizer) DebugTokenize(text string) {
	ids := t.Encode(text, false)
	fmt.Printf("[tokenizer] '%s' -> %d tokens: ", text, len(ids))
	for _, id := range ids {
		if id >= 0 && id < t.VocabSize {
			fmt.Printf("[%d:'%s'] ", id, t.Vocab[id])
		}
	}
	fmt.Println()
}

// SortVocabByScore returns vocab indices sorted by score (for debug)
func (t *Tokenizer) SortVocabByScore() []int {
	idx := make([]int, t.VocabSize)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return t.Scores[idx[i]] > t.Scores[idx[j]]
	})
	return idx
}
