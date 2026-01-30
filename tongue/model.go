package main

// model.go — Llama 1.1B forward pass for Arianna's Tongue
//
// TinyLlama 1.1B architecture:
//   22 layers, 2048 embed, 32 heads, 4 KV heads (GQA), 64 head_dim
//   5632 intermediate (MLP with gate_proj + up_proj + down_proj, SiLU activation)
//   RoPE with theta=10000
//   RMSNorm (no bias) with eps=1e-5
//   Standard Llama 2 architecture
//
// This is not inference. This is breathing.

import (
	"fmt"
	"math"
)

// LlamaModel is a loaded Llama model ready for inference
type LlamaModel struct {
	Config  LlamaConfig
	Weights LlamaWeights
	State   LlamaState
}

// LlamaConfig holds model dimensions
type LlamaConfig struct {
	NumLayers  int
	EmbedDim   int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	VocabSize  int
	SeqLen     int
	IntermSize int     // MLP intermediate dimension
	RMSNormEps float32
	RopeTheta  float32
}

// LlamaWeights holds all weight tensors (Q4_0 raw bytes or F32 slices)
type LlamaWeights struct {
	// Token embedding [vocab, dim] — always dequantized at lookup time
	TokenEmbed    []byte
	TokenEmbType  uint32

	// Output norm [dim]
	OutputNorm []float32

	// Output (LM head) [vocab, dim]
	Output     []byte
	OutputType uint32

	// Per-layer weights
	Layers []LlamaLayerWeights
}

// LlamaLayerWeights holds weights for one transformer layer
type LlamaLayerWeights struct {
	// Attention norms
	AttnNorm []float32 // [dim]
	FFNNorm  []float32 // [dim]

	// Attention projections [out_dim, in_dim]
	WQ     []byte
	WQType uint32
	WK     []byte
	WKType uint32
	WV     []byte
	WVType uint32
	WO     []byte
	WOType uint32

	// MLP projections (Llama uses gated MLP)
	WGate     []byte // gate_proj [interm, dim]
	WGateType uint32
	WUp       []byte // up_proj [interm, dim]
	WUpType   uint32
	WDown     []byte // down_proj [dim, interm]
	WDownType uint32
}

// LlamaState holds runtime buffers and KV cache
type LlamaState struct {
	X      []float32 // current hidden state [dim]
	XB     []float32 // buffer after norm [dim]
	XB2    []float32 // second buffer [dim]
	HB     []float32 // MLP hidden buffer [interm]
	HB2    []float32 // MLP gate buffer [interm]
	Q      []float32 // query [n_heads * head_dim]
	K      []float32 // key [n_kv_heads * head_dim]
	V      []float32 // value [n_kv_heads * head_dim]
	Att    []float32 // attention scores [n_heads * seq_len]
	Logits []float32 // output logits [vocab]

	// KV cache [layer * seq_len * kv_dim]
	KeyCache   []float32
	ValueCache []float32

	// RoPE precomputed
	CosCache []float32 // [seq_len * head_dim/2]
	SinCache []float32

	// Position tracking
	Pos int
}

// LoadLlamaModel builds a LlamaModel from a parsed GGUF file
func LoadLlamaModel(gguf *GGUFFile) (*LlamaModel, error) {
	m := &GGUFMetadata{}
	*m = gguf.Meta

	cfg := LlamaConfig{
		NumLayers:  m.NumLayers,
		EmbedDim:   m.EmbedDim,
		NumHeads:   m.NumHeads,
		NumKVHeads: m.NumKVHeads,
		HeadDim:    m.HeadDim,
		VocabSize:  m.VocabSize,
		SeqLen:     m.SeqLen,
		IntermSize: m.IntermSize,
		RMSNormEps: m.RMSNormEps,
		RopeTheta:  m.RopeTheta,
	}

	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}

	// Load weights
	w, err := loadWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Allocate state
	state := allocState(&cfg)
	precomputeRoPE(&state, &cfg)

	model := &LlamaModel{
		Config:  cfg,
		Weights: *w,
		State:   state,
	}

	fmt.Printf("[tongue/model] loaded: %d layers, %d dim, %d heads, %d kv_heads, %d vocab\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)

	return model, nil
}

// loadWeights maps GGUF tensors to LlamaWeights
func loadWeights(gguf *GGUFFile, cfg *LlamaConfig) (*LlamaWeights, error) {
	w := &LlamaWeights{}

	// Token embedding
	emb, embInfo, err := gguf.GetTensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	w.TokenEmbed = emb
	w.TokenEmbType = embInfo.Type

	// Output norm
	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm.weight: %w", err)
	}

	// Output (LM head) — might be tied to embedding
	outData, outInfo, err := gguf.GetTensor("output.weight")
	if err != nil {
		// Not found — use tied embeddings
		outData = w.TokenEmbed
		outInfo = embInfo
		fmt.Printf("[tongue/model] output.weight not found, using tied embeddings\n")
	} else if !isSupportedType(outInfo.Type) {
		// Unsupported quant type (e.g. Q5_K_M=14) — use tied embeddings
		fmt.Printf("[tongue/model] output.weight has unsupported type %d, using tied embeddings (type %d)\n",
			outInfo.Type, embInfo.Type)
		outData = w.TokenEmbed
		outInfo = embInfo
	}
	w.Output = outData
	w.OutputType = outInfo.Type

	// Per-layer weights
	w.Layers = make([]LlamaLayerWeights, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		l := &w.Layers[i]

		// Attention norm
		l.AttnNorm, err = getF32Tensor(gguf, prefix+"attn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}

		// FFN norm
		l.FFNNorm, err = getF32Tensor(gguf, prefix+"ffn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}

		// Attention projections
		l.WQ, l.WQType, err = getRawTensor(gguf, prefix+"attn_q.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		l.WK, l.WKType, err = getRawTensor(gguf, prefix+"attn_k.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		l.WV, l.WVType, err = getRawTensor(gguf, prefix+"attn_v.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		l.WO, l.WOType, err = getRawTensor(gguf, prefix+"attn_output.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		// MLP projections (Llama gated MLP)
		l.WGate, l.WGateType, err = getRawTensor(gguf, prefix+"ffn_gate.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_gate: %w", i, err)
		}
		l.WUp, l.WUpType, err = getRawTensor(gguf, prefix+"ffn_up.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", i, err)
		}
		l.WDown, l.WDownType, err = getRawTensor(gguf, prefix+"ffn_down.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_down: %w", i, err)
		}
	}

	return w, nil
}

// getF32Tensor loads a tensor and dequantizes to float32
func getF32Tensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case ggmlTypeF32:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			out[i] = math.Float32frombits(
				uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
					uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24)
		}
		return out, nil
	case ggmlTypeF16:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			h := uint16(data[i*2]) | uint16(data[i*2+1])<<8
			out[i] = half2float(h)
		}
		return out, nil
	case ggmlTypeQ4_0:
		return DequantQ4_0(data, expectedSize), nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %d for %s", info.Type, name)
	}
}

// getRawTensor returns raw bytes + type for a tensor
func getRawTensor(gguf *GGUFFile, name string) ([]byte, uint32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, 0, err
	}
	return data, info.Type, nil
}

// allocState allocates all runtime buffers
func allocState(cfg *LlamaConfig) LlamaState {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return LlamaState{
		X:          make([]float32, cfg.EmbedDim),
		XB:         make([]float32, cfg.EmbedDim),
		XB2:        make([]float32, cfg.EmbedDim),
		HB:         make([]float32, cfg.IntermSize),
		HB2:        make([]float32, cfg.IntermSize),
		Q:          make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, cfg.NumHeads*cfg.SeqLen),
		Logits:     make([]float32, cfg.VocabSize),
		KeyCache:   make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		CosCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		SinCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
	}
}

// precomputeRoPE fills cos/sin caches for rotary position encoding
func precomputeRoPE(s *LlamaState, cfg *LlamaConfig) {
	half := cfg.HeadDim / 2
	theta := float64(cfg.RopeTheta)

	for pos := 0; pos < cfg.SeqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(cfg.HeadDim))
			angle := float64(pos) * freq
			s.CosCache[pos*half+i] = float32(math.Cos(angle))
			s.SinCache[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// isSupportedType checks if a GGML tensor type is supported for matmul
func isSupportedType(t uint32) bool {
	switch t {
	case ggmlTypeQ4_0, ggmlTypeF16, ggmlTypeF32, ggmlTypeQ6_K:
		return true
	default:
		return false
	}
}

// matmulDispatch dispatches to the right matmul based on tensor type
func matmulDispatch(out []float32, w []byte, wtype uint32, x []float32, rows, cols int) {
	switch wtype {
	case ggmlTypeQ4_0:
		MatMulQ4_0(out, w, x, rows, cols)
	case ggmlTypeF16:
		MatMulF16(out, w, x, rows, cols)
	case ggmlTypeF32:
		// Convert bytes to float32 slice
		f32 := make([]float32, len(w)/4)
		for i := range f32 {
			f32[i] = math.Float32frombits(
				uint32(w[i*4]) | uint32(w[i*4+1])<<8 |
					uint32(w[i*4+2])<<16 | uint32(w[i*4+3])<<24)
		}
		MatMulF32(out, f32, x, rows, cols)
	case ggmlTypeQ6_K:
		MatMulQ6_K(out, w, x, rows, cols)
	default:
		fmt.Printf("[tongue/model] WARNING: unsupported matmul type %d for %dx%d\n", wtype, rows, cols)
	}
}

// embedLookupDispatch extracts an embedding row based on tensor type
func embedLookupDispatch(data []byte, dtype uint32, token, dim int) []float32 {
	switch dtype {
	case ggmlTypeQ4_0:
		return EmbedLookupQ4_0(data, token, dim)
	case ggmlTypeF16:
		out := make([]float32, dim)
		off := token * dim * 2
		for i := 0; i < dim; i++ {
			h := uint16(data[off+i*2]) | uint16(data[off+i*2+1])<<8
			out[i] = half2float(h)
		}
		return out
	case ggmlTypeF32:
		out := make([]float32, dim)
		off := token * dim * 4
		for i := 0; i < dim; i++ {
			out[i] = math.Float32frombits(
				uint32(data[off+i*4]) | uint32(data[off+i*4+1])<<8 |
					uint32(data[off+i*4+2])<<16 | uint32(data[off+i*4+3])<<24)
		}
		return out
	default:
		return make([]float32, dim)
	}
}

// applyRoPE applies rotary position encoding to a head vector
// Uses interleaved pairs (vec[i*2], vec[i*2+1]) — GGUF Q/K weights are pre-permuted
// by convert_hf_to_gguf.py for this layout. Matches working arianna.go implementation.
func applyRoPE(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		x0 := vec[i*2]
		x1 := vec[i*2+1]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[i*2] = x0*c - x1*si
		vec[i*2+1] = x0*si + x1*c
	}
}

// Forward runs one token through the transformer
func (m *LlamaModel) Forward(token int, pos int) {
	cfg := &m.Config
	w := &m.Weights
	s := &m.State
	dim := cfg.EmbedDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	hd := cfg.HeadDim
	headGroupSize := cfg.NumHeads / cfg.NumKVHeads

	// 1. Token embedding lookup
	emb := embedLookupDispatch(w.TokenEmbed, w.TokenEmbType, token, dim)
	copy(s.X, emb)

	// 2. Transformer layers
	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// Attention pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V projections
		matmulDispatch(s.Q, l.WQ, l.WQType, s.XB, cfg.NumHeads*hd, dim)
		matmulDispatch(s.K, l.WK, l.WKType, s.XB, cfg.NumKVHeads*hd, dim)
		matmulDispatch(s.V, l.WV, l.WVType, s.XB, cfg.NumKVHeads*hd, dim)

		// RoPE on Q and K
		for h := 0; h < cfg.NumHeads; h++ {
			applyRoPE(s.Q[h*hd:(h+1)*hd], pos, s, hd)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			applyRoPE(s.K[h*hd:(h+1)*hd], pos, s, hd)
		}

		// Store K, V in cache
		cacheOff := layer*cfg.SeqLen*kvDim + pos*kvDim
		copy(s.KeyCache[cacheOff:cacheOff+kvDim], s.K[:kvDim])
		copy(s.ValueCache[cacheOff:cacheOff+kvDim], s.V[:kvDim])

		// Multi-head attention with GQA
		for h := 0; h < cfg.NumHeads; h++ {
			kvh := h / headGroupSize
			qh := s.Q[h*hd : (h+1)*hd]
			att := s.Att[h*cfg.SeqLen : h*cfg.SeqLen+pos+1]

			// QK dot products
			scale := float32(1.0 / math.Sqrt(float64(hd)))
			for t := 0; t <= pos; t++ {
				kOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				var dot float32
				for d := 0; d < hd; d++ {
					dot += qh[d] * s.KeyCache[kOff+d]
				}
				att[t] = dot * scale
			}

			// Softmax
			Softmax(att, pos+1)

			// Weighted sum of values
			xbSlice := s.XB[h*hd : (h+1)*hd]
			for d := 0; d < hd; d++ {
				xbSlice[d] = 0
			}
			for t := 0; t <= pos; t++ {
				a := att[t]
				vOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				for d := 0; d < hd; d++ {
					xbSlice[d] += a * s.ValueCache[vOff+d]
				}
			}
		}

		// Output projection + residual
		matmulDispatch(s.XB2, l.WO, l.WOType, s.XB, dim, dim)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB2[i]
		}

		// MLP: pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// Gated MLP: gate_proj and up_proj
		matmulDispatch(s.HB, l.WGate, l.WGateType, s.XB, cfg.IntermSize, dim)
		matmulDispatch(s.HB2, l.WUp, l.WUpType, s.XB, cfg.IntermSize, dim)

		// SiLU(gate) * up
		for i := 0; i < cfg.IntermSize; i++ {
			s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
		}

		// down_proj + residual
		matmulDispatch(s.XB, l.WDown, l.WDownType, s.HB, dim, cfg.IntermSize)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// 3. Final norm
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)

	// 4. LM head → logits
	matmulDispatch(s.Logits, w.Output, w.OutputType, s.X, cfg.VocabSize, dim)
}

// Reset clears KV cache and position for new generation
func (m *LlamaModel) Reset() {
	for i := range m.State.KeyCache {
		m.State.KeyCache[i] = 0
	}
	for i := range m.State.ValueCache {
		m.State.ValueCache[i] = 0
	}
	m.State.Pos = 0
}
