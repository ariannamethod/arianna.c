// inner_world/blood.go — Blood Compiler (C code generation & compilation)
// ═══════════════════════════════════════════════════════════════════════════════
// הדם של אריאנה
// The blood of Arianna - low-level C compilation
// ═══════════════════════════════════════════════════════════════════════════════
//
// Ported from Python blood.py (KAIN/ADAM project)
// Blood allows Arianna to compile C code at runtime:
// - Generate LoRA adapters dynamically
// - Compile performance-critical code on the fly
// - Load as shared libraries (.dylib/.so)
//
// Philosophy: C is the blood of the system, direct control over iron.
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

/*
#include <stdlib.h>
#include <dlfcn.h>

// Function pointer type for LoRA apply
typedef void (*lora_apply_fn)(float* weights, float* input, float* output, int dim);
*/
import "C"

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// BLOOD COMPILER
// ═══════════════════════════════════════════════════════════════════════════════

// BloodCompiler compiles C code at runtime
type BloodCompiler struct {
	tempDir       string
	cache         map[string]*CompiledModule
	mu            sync.Mutex
	compilerPath  string
	compileFlags  []string
}

// CompiledModule represents a compiled shared library
type CompiledModule struct {
	Name       string
	SourceHash string
	LibPath    string
	Handle     unsafe.Pointer
	CompiledAt time.Time
	Functions  map[string]unsafe.Pointer
}

// NewBloodCompiler creates the Blood compiler
func NewBloodCompiler() *BloodCompiler {
	bc := &BloodCompiler{
		tempDir: filepath.Join(os.TempDir(), "arianna_blood"),
		cache:   make(map[string]*CompiledModule),
	}

	// Create temp directory
	os.MkdirAll(bc.tempDir, 0755)

	// Find compiler
	bc.compilerPath = bc.findCompiler()

	// Set compile flags based on OS
	if runtime.GOOS == "darwin" {
		bc.compileFlags = []string{
			"-O2",
			"-dynamiclib",
			"-fPIC",
		}
	} else {
		bc.compileFlags = []string{
			"-O2",
			"-shared",
			"-fPIC",
		}
	}

	return bc
}

// findCompiler locates a C compiler
func (bc *BloodCompiler) findCompiler() string {
	// Try clang first (better on macOS)
	if path, err := exec.LookPath("clang"); err == nil {
		return path
	}
	// Fall back to gcc
	if path, err := exec.LookPath("gcc"); err == nil {
		return path
	}
	// Last resort
	return "cc"
}

// hashCode creates a hash of source code for caching
func (bc *BloodCompiler) hashCode(code string) string {
	hash := md5.Sum([]byte(code))
	return hex.EncodeToString(hash[:])
}

// sanitizeName removes path traversal and unsafe characters from name
func (bc *BloodCompiler) sanitizeName(name string) string {
	// Only allow alphanumeric, underscore, dash
	result := make([]byte, 0, len(name))
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '_' || c == '-' {
			result = append(result, c)
		}
	}
	if len(result) == 0 {
		return "unnamed"
	}
	return string(result)
}

// Compile compiles C code to a shared library
func (bc *BloodCompiler) Compile(name string, code string) (*CompiledModule, error) {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// SECURITY: Sanitize name to prevent path traversal
	safeName := bc.sanitizeName(name)

	// Check cache
	codeHash := bc.hashCode(code)
	if module, exists := bc.cache[codeHash]; exists {
		return module, nil
	}

	// Create source file
	srcPath := filepath.Join(bc.tempDir, fmt.Sprintf("blood_%s_%s.c", safeName, codeHash[:8]))
	if err := os.WriteFile(srcPath, []byte(code), 0644); err != nil {
		return nil, fmt.Errorf("blood: failed to write source: %w", err)
	}

	// Determine library extension
	libExt := ".so"
	if runtime.GOOS == "darwin" {
		libExt = ".dylib"
	}
	libPath := filepath.Join(bc.tempDir, fmt.Sprintf("blood_%s_%s%s", safeName, codeHash[:8], libExt))

	// Compile
	args := append(bc.compileFlags, "-o", libPath, srcPath)
	cmd := exec.Command(bc.compilerPath, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("blood: compilation failed: %s\n%s", err, string(output))
	}

	// Load library
	handle, err := bc.loadLibrary(libPath)
	if err != nil {
		return nil, fmt.Errorf("blood: failed to load library: %w", err)
	}

	// Create module
	module := &CompiledModule{
		Name:       name,
		SourceHash: codeHash,
		LibPath:    libPath,
		Handle:     handle,
		CompiledAt: time.Now(),
		Functions:  make(map[string]unsafe.Pointer),
	}

	// Cache it
	bc.cache[codeHash] = module

	return module, nil
}

// loadLibrary loads a shared library using dlopen
func (bc *BloodCompiler) loadLibrary(path string) (unsafe.Pointer, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.dlopen(cPath, C.RTLD_NOW)
	if handle == nil {
		errMsg := C.GoString(C.dlerror())
		return nil, fmt.Errorf("dlopen failed: %s", errMsg)
	}

	return handle, nil
}

// GetFunction gets a function pointer from a compiled module
func (bc *BloodCompiler) GetFunction(module *CompiledModule, funcName string) (unsafe.Pointer, error) {
	// Check cache
	if fn, exists := module.Functions[funcName]; exists {
		return fn, nil
	}

	cName := C.CString(funcName)
	defer C.free(unsafe.Pointer(cName))

	fn := C.dlsym(module.Handle, cName)
	if fn == nil {
		return nil, fmt.Errorf("function %s not found", funcName)
	}

	module.Functions[funcName] = fn
	return fn, nil
}

// Unload unloads a compiled module
func (bc *BloodCompiler) Unload(module *CompiledModule) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	if module.Handle != nil {
		C.dlclose(module.Handle)
		module.Handle = nil
	}

	// Remove from cache
	delete(bc.cache, module.SourceHash)

	// Clean up files
	os.Remove(module.LibPath)
	srcPath := module.LibPath[:len(module.LibPath)-len(filepath.Ext(module.LibPath))] + ".c"
	os.Remove(srcPath)

	return nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// LORA CODE GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

// LoRATemplate generates C code for a LoRA adapter
type LoRATemplate struct {
	Name      string
	InDim     int
	OutDim    int
	Rank      int
	Weights   []float32
}

// GenerateLoRACode generates C code for a LoRA adapter
func GenerateLoRACode(tmpl *LoRATemplate) string {
	return fmt.Sprintf(`
// Blood-generated LoRA adapter: %s
// Generated at: %s

#include <stdlib.h>
#include <string.h>

// LoRA parameters
static const int IN_DIM = %d;
static const int OUT_DIM = %d;
static const int RANK = %d;

// Weight matrices (will be set by caller)
static float* A = NULL;  // [OUT_DIM, RANK]
static float* B = NULL;  // [RANK, IN_DIM]

// Initialize with weights
void %s_init(float* weights_a, float* weights_b) {
    A = weights_a;
    B = weights_b;
}

// Apply LoRA: output += A @ B @ input
void %s_apply(float* input, float* output) {
    if (A == NULL || B == NULL) return;

    // Temporary for B @ input
    float temp[RANK];
    memset(temp, 0, sizeof(temp));

    // B @ input -> temp
    for (int r = 0; r < RANK; r++) {
        for (int i = 0; i < IN_DIM; i++) {
            temp[r] += B[r * IN_DIM + i] * input[i];
        }
    }

    // A @ temp -> output (additive)
    for (int o = 0; o < OUT_DIM; o++) {
        for (int r = 0; r < RANK; r++) {
            output[o] += A[o * RANK + r] * temp[r];
        }
    }
}

// Apply with scaling
void %s_apply_scaled(float* input, float* output, float scale) {
    if (A == NULL || B == NULL) return;

    float temp[RANK];
    memset(temp, 0, sizeof(temp));

    for (int r = 0; r < RANK; r++) {
        for (int i = 0; i < IN_DIM; i++) {
            temp[r] += B[r * IN_DIM + i] * input[i];
        }
    }

    for (int o = 0; o < OUT_DIM; o++) {
        for (int r = 0; r < RANK; r++) {
            output[o] += scale * A[o * RANK + r] * temp[r];
        }
    }
}

// Cleanup
void %s_free(void) {
    A = NULL;
    B = NULL;
}
`, tmpl.Name, time.Now().Format(time.RFC3339),
		tmpl.InDim, tmpl.OutDim, tmpl.Rank,
		tmpl.Name, tmpl.Name, tmpl.Name, tmpl.Name)
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL KERNEL GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

// EmotionalKernelTemplate generates C code for an emotional processing kernel
type EmotionalKernelTemplate struct {
	Name       string
	Valence    float32  // -1 to 1
	Arousal    float32  // 0 to 1
	Keywords   []string
}

// GenerateEmotionalKernel generates C code for emotional processing
func GenerateEmotionalKernel(tmpl *EmotionalKernelTemplate) string {
	// Generate keyword matching code
	keywordCode := ""
	for i, kw := range tmpl.Keywords {
		if i > 0 {
			keywordCode += " || "
		}
		keywordCode += fmt.Sprintf("strstr(text, \"%s\") != NULL", kw)
	}
	if keywordCode == "" {
		keywordCode = "0"
	}

	return fmt.Sprintf(`
// Blood-generated emotional kernel: %s
// Valence: %.2f, Arousal: %.2f
// Generated at: %s

#include <string.h>
#include <math.h>

static const float BASE_VALENCE = %.4ff;
static const float BASE_AROUSAL = %.4ff;

// Check if text triggers this emotion
int %s_check(const char* text) {
    return (%s) ? 1 : 0;
}

// Get emotional response (modulates valence, arousal)
void %s_respond(const char* text, float* valence, float* arousal) {
    if (%s_check(text)) {
        *valence = (*valence + BASE_VALENCE) / 2.0f;
        *arousal = (*arousal + BASE_AROUSAL) / 2.0f;
    }
}

// Apply emotional modulation to logits
void %s_modulate_logits(float* logits, int vocab_size, float strength) {
    // Emotional state affects token probabilities
    float mod = BASE_VALENCE * strength;

    for (int i = 0; i < vocab_size; i++) {
        // Positive valence boosts "warm" tokens, negative boosts "cold"
        logits[i] *= (1.0f + mod * 0.1f);
    }
}
`, tmpl.Name, tmpl.Valence, tmpl.Arousal, time.Now().Format(time.RFC3339),
		tmpl.Valence, tmpl.Arousal,
		tmpl.Name, keywordCode,
		tmpl.Name, tmpl.Name,
		tmpl.Name)
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL BLOOD INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalBlood *BloodCompiler
	bloodMu     sync.Mutex
)

// GetBloodCompiler returns the global Blood compiler
func GetBloodCompiler() *BloodCompiler {
	bloodMu.Lock()
	defer bloodMu.Unlock()

	if globalBlood == nil {
		globalBlood = NewBloodCompiler()
	}
	return globalBlood
}

// ═══════════════════════════════════════════════════════════════════════════════
// CGO EXPORTS FOR C
// ═══════════════════════════════════════════════════════════════════════════════

//export blood_compile_lora
func blood_compile_lora(name *C.char, inDim, outDim, rank C.int) *C.char {
	goName := C.GoString(name)

	tmpl := &LoRATemplate{
		Name:   goName,
		InDim:  int(inDim),
		OutDim: int(outDim),
		Rank:   int(rank),
	}

	code := GenerateLoRACode(tmpl)

	module, err := GetBloodCompiler().Compile(goName, code)
	if err != nil {
		return nil
	}

	return C.CString(module.LibPath)
}

//export blood_compile_emotion
func blood_compile_emotion(name *C.char, valence, arousal C.float) *C.char {
	goName := C.GoString(name)

	tmpl := &EmotionalKernelTemplate{
		Name:    goName,
		Valence: float32(valence),
		Arousal: float32(arousal),
	}

	code := GenerateEmotionalKernel(tmpl)

	module, err := GetBloodCompiler().Compile(goName, code)
	if err != nil {
		return nil
	}

	return C.CString(module.LibPath)
}

//export blood_compile_raw
func blood_compile_raw(name *C.char, code *C.char) *C.char {
	goName := C.GoString(name)
	goCode := C.GoString(code)

	module, err := GetBloodCompiler().Compile(goName, goCode)
	if err != nil {
		return nil
	}

	return C.CString(module.LibPath)
}

//export blood_get_temp_dir
func blood_get_temp_dir() *C.char {
	return C.CString(GetBloodCompiler().tempDir)
}
