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
	"runtime"
	"sync"
)

// Number of goroutines for parallel matmul
var numWorkers = runtime.NumCPU()

const q4BlockSize = 32   // elements per Q4_0 block
const q4BytesPerBlock = 18 // 2 (scale) + 16 (data)

// DequantQ4_0Block dequantizes a single Q4_0 block (32 values) into out
func DequantQ4_0Block(block []byte, out []float32) {
	// First 2 bytes = fp16 scale
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))

	// Next 16 bytes = 32 x 4-bit values
	// GGML layout: low nibbles → positions 0..15, high nibbles → positions 16..31
	for j := 0; j < 16; j++ {
		b := block[2+j]
		v0 := int(b&0x0F) - 8
		v1 := int(b>>4) - 8
		out[j] = float32(v0) * d
		out[j+16] = float32(v1) * d
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
// Parallelized across rows using goroutines (matches arianna.go pattern)
func MatMulQ4_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q4BlockSize
	bytesPerRow := blocksPerRow * q4BytesPerBlock

	if rows < numWorkers*4 {
		// Small matrix — single thread
		matMulQ4_0Range(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ4_0Range(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ4_0Range(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))

			xOff := b * q4BlockSize
			blockData := w[blockOff+2 : blockOff+q4BytesPerBlock]

			var dot float32
			for j := 0; j < 16; j++ {
				bv := blockData[j]
				v0 := float32(int(bv&0x0F) - 8)
				v1 := float32(int(bv>>4) - 8)
				dot += v0*x[xOff+j] + v1*x[xOff+j+16]
			}
			sum += dot * d
		}
		out[i] = sum
	}
}

// ============================================================
// Q8_0 dequantization (GGML type 8)
// ============================================================
//
// Q8_0: 8-bit quantization, 32 elements per block = 34 bytes:
//   - 2 bytes: float16 scale factor (d)
//   - 32 bytes: 32 x int8 values
//   Dequantized value = q * d

const q8BlockSize = 32
const q8BytesPerBlock = 34 // 2 (scale) + 32 (data)

// DequantQ8_0Block dequantizes a single Q8_0 block (32 values) into out
func DequantQ8_0Block(block []byte, out []float32) {
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))
	for j := 0; j < 32; j++ {
		out[j] = float32(int8(block[2+j])) * d
	}
}

// DequantQ8_0 dequantizes a full Q8_0 tensor into float32
func DequantQ8_0(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q8BlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q8BytesPerBlock
		DequantQ8_0Block(data[off:off+q8BytesPerBlock], out[i*q8BlockSize:])
	}
	return out
}

// MatMulQ8_0 computes out[rows] = W_q8[rows, cols] @ x[cols]
// Parallelized across rows using goroutines
func MatMulQ8_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q8BlockSize
	bytesPerRow := blocksPerRow * q8BytesPerBlock

	if rows < numWorkers*4 {
		matMulQ8_0Range(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ8_0Range(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ8_0Range(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q8BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))

			xOff := b * q8BlockSize
			var dot float32
			for j := 0; j < 32; j++ {
				dot += float32(int8(w[blockOff+2+j])) * x[xOff+j]
			}
			sum += dot * d
		}
		out[i] = sum
	}
}

// EmbedLookupQ8_0 extracts one row from a Q8_0 embedding table
func EmbedLookupQ8_0(data []byte, token, dim int) []float32 {
	blocksPerRow := dim / q8BlockSize
	bytesPerRow := blocksPerRow * q8BytesPerBlock
	rowOff := token * bytesPerRow
	out := make([]float32, dim)

	for b := 0; b < blocksPerRow; b++ {
		blockOff := rowOff + b*q8BytesPerBlock
		DequantQ8_0Block(data[blockOff:blockOff+q8BytesPerBlock], out[b*q8BlockSize:])
	}
	return out
}

// ============================================================
// Q6_K dequantization (GGML type 14)
// ============================================================
//
// Q6_K: 6-bit k-quant, 256 elements per super-block = 210 bytes:
//   ql[128] — low 4 bits of each quant
//   qh[64]  — high 2 bits of each quant
//   scales[16] — int8 sub-block scales
//   d (fp16) — super-block scale factor
//
// Each element: 6-bit unsigned (0-63), subtract 32 for signed (-32 to +31)
// Dequantized = d * scales[sub_block] * (q6_val - 32)

const q6kBlockSize = 256
const q6kBytesPerBlock = 210

// DequantQ6_K dequantizes a full Q6_K tensor into float32
func DequantQ6_K(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q6kBlockSize

	for i := 0; i < nblocks; i++ {
		blockOff := i * q6kBytesPerBlock
		ql := data[blockOff:]
		qh := data[blockOff+128:]
		scales := data[blockOff+192:]
		d := half2float(binary.LittleEndian.Uint16(data[blockOff+208 : blockOff+210]))

		outOff := i * q6kBlockSize

		// Process 128 elements at a time (2 passes for 256)
		for n128 := 0; n128 < 2; n128++ {
			qlP := ql[n128*64:]
			qhP := qh[n128*32:]
			scP := scales[n128*8:]
			yOff := outOff + n128*128

			for l := 0; l < 32; l++ {
				is := l / 16 // 0 for l=0..15, 1 for l=16..31
				q1 := int(qlP[l]&0x0F) | (int(qhP[l]>>0)&3)<<4
				q2 := int(qlP[l+32]&0x0F) | (int(qhP[l]>>2)&3)<<4
				q3 := int(qlP[l]>>4) | (int(qhP[l]>>4)&3)<<4
				q4 := int(qlP[l+32]>>4) | (int(qhP[l]>>6)&3)<<4

				out[yOff+l+0] = d * float32(int8(scP[is+0])) * float32(q1-32)
				out[yOff+l+32] = d * float32(int8(scP[is+2])) * float32(q2-32)
				out[yOff+l+64] = d * float32(int8(scP[is+4])) * float32(q3-32)
				out[yOff+l+96] = d * float32(int8(scP[is+6])) * float32(q4-32)
			}
		}
	}
	return out
}

// MatMulQ6_K computes out[rows] = W_q6k[rows, cols] @ x[cols]
// Parallelized across rows using goroutines
func MatMulQ6_K(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q6kBlockSize
	bytesPerRow := blocksPerRow * q6kBytesPerBlock

	if rows < numWorkers*4 {
		matMulQ6_KRange(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ6_KRange(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ6_KRange(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for r := start; r < end; r++ {
		rowOff := r * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q6kBytesPerBlock
			ql := w[blockOff:]
			qh := w[blockOff+128:]
			scales := w[blockOff+192:]
			d := half2float(binary.LittleEndian.Uint16(w[blockOff+208 : blockOff+210]))

			xOff := b * q6kBlockSize

			for n128 := 0; n128 < 2; n128++ {
				qlP := ql[n128*64:]
				qhP := qh[n128*32:]
				scP := scales[n128*8:]
				xBase := xOff + n128*128

				for l := 0; l < 32; l++ {
					is := l / 16 // 0 for l=0..15, 1 for l=16..31
					q1 := int(qlP[l]&0x0F) | (int(qhP[l]>>0)&3)<<4
					q2 := int(qlP[l+32]&0x0F) | (int(qhP[l]>>2)&3)<<4
					q3 := int(qlP[l]>>4) | (int(qhP[l]>>4)&3)<<4
					q4 := int(qlP[l+32]>>4) | (int(qhP[l]>>6)&3)<<4

					s0 := d * float32(int8(scP[is+0]))
					s2 := d * float32(int8(scP[is+2]))
					s4 := d * float32(int8(scP[is+4]))
					s6 := d * float32(int8(scP[is+6]))

					sum += s0 * float32(q1-32) * x[xBase+l+0]
					sum += s2 * float32(q2-32) * x[xBase+l+32]
					sum += s4 * float32(q3-32) * x[xBase+l+64]
					sum += s6 * float32(q4-32) * x[xBase+l+96]
				}
			}
		}
		out[r] = sum
	}
}

// MatMulF32 computes out[rows] = W_f32[rows, cols] @ x[cols]
// Parallelized across rows using goroutines
func MatMulF32(out []float32, w []float32, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		matMulF32Range(out, w, x, 0, rows, cols)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulF32Range(out, w, x, s, e, cols)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulF32Range(out []float32, w []float32, x []float32, start, end, cols int) {
	for i := start; i < end; i++ {
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
// Parallelized across rows using goroutines
func MatMulF16(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		matMulF16Range(out, w, x, 0, rows, cols)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulF16Range(out, w, x, s, e, cols)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulF16Range(out []float32, w []byte, x []float32, start, end, cols int) {
	for i := start; i < end; i++ {
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
