package main

// quant.go — Q4_0 dequantization and quantized matrix operations
//
// GGML Q4_0 format:
//   Block of 32 elements = 18 bytes:
//     - 2 bytes: float16 scale factor (d)
//     - 16 bytes: 32 x 4-bit unsigned values packed in pairs
//   Each 4-bit value is unsigned [0..15], subtract 8 to get signed [-8..7]
//   Dequantized value = (q - 8) * d
//
// Memory layout per block:
//   [d_fp16] [q0q1] [q2q3] ... [q30q31]
//    2 bytes   1      1    ...    1     = 18 bytes total

import (
	"encoding/binary"
	"math"
)

const q4BlockSize = 32   // elements per Q4_0 block
const q4BytesPerBlock = 18 // 2 (scale) + 16 (data)

// DequantQ4_0Block dequantizes a single Q4_0 block (32 values) into out
func DequantQ4_0Block(block []byte, out []float32) {
	// First 2 bytes = fp16 scale
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))

	// Next 16 bytes = 32 x 4-bit values
	for j := 0; j < 16; j++ {
		b := block[2+j]
		v0 := int(b&0x0F) - 8
		v1 := int(b>>4) - 8
		out[j*2] = float32(v0) * d
		out[j*2+1] = float32(v1) * d
	}
}

// DequantQ4_0 dequantizes a full Q4_0 tensor into float32
func DequantQ4_0(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q4BlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q4BytesPerBlock
		DequantQ4_0Block(data[off:off+q4BytesPerBlock], out[i*q4BlockSize:])
	}
	return out
}

// MatMulQ4_0 computes out[rows] = W_q4[rows, cols] @ x[cols]
// W is stored as Q4_0 blocks in row-major order
func MatMulQ4_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q4BlockSize
	bytesPerRow := blocksPerRow * q4BytesPerBlock

	for i := 0; i < rows; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))

			xOff := b * q4BlockSize
			blockData := w[blockOff+2 : blockOff+q4BytesPerBlock]

			// Unrolled dot product over 32 elements
			var dot float32
			for j := 0; j < 16; j++ {
				bv := blockData[j]
				v0 := float32(int(bv&0x0F) - 8)
				v1 := float32(int(bv>>4) - 8)
				dot += v0*x[xOff+j*2] + v1*x[xOff+j*2+1]
			}
			sum += dot * d
		}
		out[i] = sum
	}
}

// MatMulF32 computes out[rows] = W_f32[rows, cols] @ x[cols]
func MatMulF32(out []float32, w []float32, x []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		sum := float32(0)
		off := i * cols
		for j := 0; j < cols; j++ {
			sum += w[off+j] * x[j]
		}
		out[i] = sum
	}
}

// MatMulF16 computes out[rows] = W_f16[rows, cols] @ x[cols]
// w is raw bytes of float16 values
func MatMulF16(out []float32, w []byte, x []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		sum := float32(0)
		rowOff := i * cols * 2
		for j := 0; j < cols; j++ {
			wv := half2float(binary.LittleEndian.Uint16(w[rowOff+j*2 : rowOff+j*2+2]))
			sum += wv * x[j]
		}
		out[i] = sum
	}
}

// EmbedLookupQ4_0 extracts one row from a Q4_0 embedding table
func EmbedLookupQ4_0(data []byte, token, dim int) []float32 {
	blocksPerRow := dim / q4BlockSize
	bytesPerRow := blocksPerRow * q4BytesPerBlock
	rowOff := token * bytesPerRow
	out := make([]float32, dim)

	for b := 0; b < blocksPerRow; b++ {
		blockOff := rowOff + b*q4BytesPerBlock
		DequantQ4_0Block(data[blockOff:blockOff+q4BytesPerBlock], out[b*q4BlockSize:])
	}
	return out
}

// EmbedLookupF32 extracts one row from an F32 embedding table
func EmbedLookupF32(data []float32, token, dim int) []float32 {
	out := make([]float32, dim)
	copy(out, data[token*dim:(token+1)*dim])
	return out
}

// RMSNorm applies RMS normalization in-place
func RMSNorm(x []float32, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * inv * w[i]
	}
}

// RMSNormInto applies RMS normalization: out = norm(x) * w
func RMSNormInto(out, x, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * inv * w[i]
	}
}

// Softmax computes softmax in-place over x[0:n]
func Softmax(x []float32, n int) {
	max := x[0]
	for i := 1; i < n; i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// SiLU activation: x * sigmoid(x)
func SiLU(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}
