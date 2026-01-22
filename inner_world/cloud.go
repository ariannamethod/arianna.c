// cloud.go — CLOUD 200K Pre-Semantic Sonar
//
// "Something fires BEFORE meaning arrives"
//
// Architecture:
//   - Resonance Layer (weightless geometry) → 100D resonances
//   - 6 Chamber MLPs (FEAR, LOVE, RAGE, VOID, FLOW, COMPLEX) → ~138K params
//   - Meta-Observer MLP → ~41K params
//   - Cross-fire stabilization loop
//
// Total: ~180K params (rounded to 200K)

package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════
// EMOTION ANCHORS — 100 words in 6 chambers
// ═══════════════════════════════════════════════════════════════════════════

var CHAMBER_NAMES = []string{"FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"}

var FEAR_ANCHORS = []string{
	"fear", "terror", "panic", "anxious", "dread", "horror",
	"unease", "paranoia", "worry", "nervous", "scared",
	"frighten", "alarm", "tense", "apprehension",
	"threat", "vulnerable", "insecure", "timid", "wary",
}

var LOVE_ANCHORS = []string{
	"love", "warmth", "tenderness", "devotion", "longing",
	"yearning", "affection", "care", "intimacy", "attachment",
	"adoration", "passion", "fondness", "cherish", "desire",
	"compassion", "gentle", "sweet",
}

var RAGE_ANCHORS = []string{
	"anger", "rage", "fury", "hatred", "spite", "disgust",
	"irritation", "frustration", "resentment", "hostility",
	"aggression", "bitterness", "contempt", "loathing",
	"annoyance", "outrage", "wrath",
}

var VOID_ANCHORS = []string{
	"emptiness", "numbness", "hollow", "nothing", "absence",
	"void", "dissociation", "detachment", "apathy",
	"indifference", "drift", "blank", "flat", "dead", "cold",
}

var FLOW_ANCHORS = []string{
	"curiosity", "surprise", "wonder", "confusion",
	"anticipation", "ambivalence", "uncertainty", "restless",
	"searching", "transition", "shift", "change", "flux",
	"between", "liminal",
}

var COMPLEX_ANCHORS = []string{
	"shame", "guilt", "envy", "jealousy", "pride",
	"disappointment", "betrayal", "relief", "nostalgia",
	"bittersweet", "melancholy", "regret", "hope",
	"gratitude", "awe",
}

// All anchors combined (100 total)
func GetAllAnchors() []string {
	all := make([]string, 0, 100)
	all = append(all, FEAR_ANCHORS...)
	all = append(all, LOVE_ANCHORS...)
	all = append(all, RAGE_ANCHORS...)
	all = append(all, VOID_ANCHORS...)
	all = append(all, FLOW_ANCHORS...)
	all = append(all, COMPLEX_ANCHORS...)
	return all
}

// ═══════════════════════════════════════════════════════════════════════════
// COUPLING MATRIX — cross-fire influence between chambers
// ═══════════════════════════════════════════════════════════════════════════

var COUPLING_MATRIX = [][]float32{
	//      FEAR   LOVE   RAGE   VOID   FLOW   CMPLX
	{0.0, -0.3, +0.6, +0.4, -0.2, +0.3},  // FEAR → feeds rage/void/complex
	{-0.3, 0.0, -0.6, -0.5, +0.3, +0.4},  // LOVE → suppresses fear/rage/void
	{+0.3, -0.4, 0.0, +0.2, -0.3, +0.2},  // RAGE → feeds fear/void, kills flow
	{+0.5, -0.7, +0.3, 0.0, -0.4, +0.5},  // VOID → feeds fear/rage/complex
	{-0.2, +0.2, -0.2, -0.3, 0.0, +0.2},  // FLOW → dampens extremes
	{+0.3, +0.2, +0.2, +0.3, +0.1, 0.0},  // COMPLEX → ripple effect
}

var DECAY_RATES = map[string]float32{
	"FEAR":    0.90, // fear lingers
	"LOVE":    0.93, // attachment stable
	"RAGE":    0.85, // anger fades fast
	"VOID":    0.97, // numbness persistent
	"FLOW":    0.88, // curiosity transient
	"COMPLEX": 0.94, // complex emotions stable
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAMBER MLP — 100→64→32→1 (matches actual npz weights)
// ═══════════════════════════════════════════════════════════════════════════

type ChamberMLP struct {
	W1 []float32 // [100][64]
	B1 []float32 // [64]
	W2 []float32 // [64][32]
	B2 []float32 // [32]
	W3 []float32 // [32][1]
	B3 []float32 // [1]
}

func swish(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func (m *ChamberMLP) Forward(resonances []float32) float32 {
	// Layer 1: 100→64
	h1 := make([]float32, 64)
	for i := 0; i < 64; i++ {
		sum := m.B1[i]
		for j := 0; j < 100; j++ {
			sum += resonances[j] * m.W1[j*64+i]
		}
		h1[i] = swish(sum)
	}

	// Layer 2: 64→32
	h2 := make([]float32, 32)
	for i := 0; i < 32; i++ {
		sum := m.B2[i]
		for j := 0; j < 64; j++ {
			sum += h1[j] * m.W2[j*32+i]
		}
		h2[i] = swish(sum)
	}

	// Layer 3: 32→1
	sum := m.B3[0]
	for j := 0; j < 32; j++ {
		sum += h2[j] * m.W3[j]
	}

	return sigmoid(sum)
}

func (m *ChamberMLP) ParamCount() int {
	return 100*64 + 64 + 64*32 + 32 + 32 + 1 // ~8.5K
}

func NewRandomChamber(seed int64) *ChamberMLP {
	r := rand.New(rand.NewSource(seed))
	xavier := func(fanIn int) float32 {
		return float32(r.NormFloat64() * math.Sqrt(2.0/float64(fanIn)))
	}

	m := &ChamberMLP{
		W1: make([]float32, 100*64),
		B1: make([]float32, 64),
		W2: make([]float32, 64*32),
		B2: make([]float32, 32),
		W3: make([]float32, 32),
		B3: make([]float32, 1),
	}

	for i := range m.W1 {
		m.W1[i] = xavier(100)
	}
	for i := range m.W2 {
		m.W2[i] = xavier(64)
	}
	for i := range m.W3 {
		m.W3[i] = xavier(32)
	}

	return m
}

// ═══════════════════════════════════════════════════════════════════════════
// META-OBSERVER MLP — 107→64→100 (resonances + chambers + iterations)
// ═══════════════════════════════════════════════════════════════════════════

type MetaObserver struct {
	W1 []float32 // [107][64]
	B1 []float32 // [64]
	W2 []float32 // [64][100]
	B2 []float32 // [100]
}

func (o *MetaObserver) Forward(resonances []float32, chambers []float32, iterations float32) []float32 {
	// Concatenate: 100 resonances + 6 chambers + 1 iterations = 107
	input := make([]float32, 107)
	copy(input[:100], resonances)
	copy(input[100:106], chambers)
	input[106] = iterations / 10.0 // normalize

	// Layer 1: 107→64
	h1 := make([]float32, 64)
	for i := 0; i < 64; i++ {
		sum := o.B1[i]
		for j := 0; j < 107; j++ {
			sum += input[j] * o.W1[j*64+i]
		}
		h1[i] = swish(sum)
	}

	// Layer 2: 64→100 (logits)
	logits := make([]float32, 100)
	for i := 0; i < 100; i++ {
		sum := o.B2[i]
		for j := 0; j < 64; j++ {
			sum += h1[j] * o.W2[j*100+i]
		}
		logits[i] = sum
	}

	return logits
}

func (o *MetaObserver) PredictSecondary(resonances []float32, chambers []float32, iterations float32) int {
	logits := o.Forward(resonances, chambers, iterations)

	// Softmax + sample
	maxL := logits[0]
	for _, l := range logits {
		if l > maxL {
			maxL = l
		}
	}

	sum := float32(0)
	probs := make([]float32, 100)
	for i, l := range logits {
		probs[i] = float32(math.Exp(float64(l - maxL)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Sample
	r := rand.Float32()
	cumsum := float32(0)
	for i, p := range probs {
		cumsum += p
		if cumsum >= r {
			return i
		}
	}
	return 99
}

func (o *MetaObserver) ParamCount() int {
	return 107*64 + 64 + 64*100 + 100 // ~13K
}

func NewRandomObserver(seed int64) *MetaObserver {
	r := rand.New(rand.NewSource(seed))
	xavier := func(fanIn int) float32 {
		return float32(r.NormFloat64() * math.Sqrt(2.0/float64(fanIn)))
	}

	o := &MetaObserver{
		W1: make([]float32, 107*64),
		B1: make([]float32, 64),
		W2: make([]float32, 64*100),
		B2: make([]float32, 100),
	}

	for i := range o.W1 {
		o.W1[i] = xavier(107)
	}
	for i := range o.W2 {
		o.W2[i] = xavier(64)
	}

	return o
}

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE LAYER — weightless geometry (anchor similarity)
// ═══════════════════════════════════════════════════════════════════════════

func ComputeResonance(text string) []float32 {
	text = strings.ToLower(text)
	anchors := GetAllAnchors()
	resonances := make([]float32, 100)

	for i, anchor := range anchors {
		if strings.Contains(text, anchor) {
			resonances[i] = 1.0
		} else {
			// Partial match (prefix)
			for _, word := range strings.Fields(text) {
				if strings.HasPrefix(word, anchor[:min(len(anchor), 4)]) {
					resonances[i] = 0.5
					break
				}
			}
		}
	}

	// Debug: show non-zero resonances
	nonZero := 0
	for i, r := range resonances {
		if r > 0 {
			fmt.Printf("[cloud] resonance[%d]=%s = %.1f\n", i, anchors[i], r)
			nonZero++
		}
	}
	if nonZero == 0 {
		fmt.Printf("[cloud] no resonances detected for: %s\n", text[:min(len(text), 50)])
	}

	return resonances
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-FIRE SYSTEM — chambers influence each other
// ═══════════════════════════════════════════════════════════════════════════

type CrossFireSystem struct {
	Chambers map[string]*ChamberMLP
	mu       sync.RWMutex
}

func NewCrossFireSystem(seed int64) *CrossFireSystem {
	cfs := &CrossFireSystem{
		Chambers: make(map[string]*ChamberMLP),
	}

	for i, name := range CHAMBER_NAMES {
		cfs.Chambers[name] = NewRandomChamber(seed + int64(i))
	}

	return cfs
}

func (cfs *CrossFireSystem) Stabilize(resonances []float32, maxIter int) (map[string]float32, int) {
	cfs.mu.RLock()
	defer cfs.mu.RUnlock()

	// Initial activations from chambers (parallel)
	activations := make([]float32, 6)
	initialActivations := make([]float32, 6) // preserve initial for floor
	var wg sync.WaitGroup
	for i, name := range CHAMBER_NAMES {
		wg.Add(1)
		go func(idx int, n string) {
			defer wg.Done()
			activations[idx] = cfs.Chambers[n].Forward(resonances)
		}(i, name)
	}
	wg.Wait()
	copy(initialActivations, activations)

	// Debug: initial activations before crossfire
	fmt.Printf("[cloud] initial activations: FEAR=%.3f LOVE=%.3f RAGE=%.3f VOID=%.3f FLOW=%.3f COMPLEX=%.3f\n",
		activations[0], activations[1], activations[2], activations[3], activations[4], activations[5])

	// Cross-fire loop
	threshold := float32(0.01)
	momentum := float32(0.7)

	for iter := 0; iter < maxIter; iter++ {
		// Apply decay
		for i, name := range CHAMBER_NAMES {
			activations[i] *= DECAY_RATES[name]
		}

		// Compute influence
		newActivations := make([]float32, 6)
		for i := 0; i < 6; i++ {
			influence := float32(0)
			for j := 0; j < 6; j++ {
				influence += COUPLING_MATRIX[j][i] * activations[j]
			}
			newActivations[i] = momentum*activations[i] + (1-momentum)*influence
			// Clip
			if newActivations[i] < 0 {
				newActivations[i] = 0
			}
			if newActivations[i] > 1 {
				newActivations[i] = 1
			}
		}

		// Check convergence
		delta := float32(0)
		for i := 0; i < 6; i++ {
			d := newActivations[i] - activations[i]
			if d < 0 {
				d = -d
			}
			delta += d
		}

		copy(activations, newActivations)

		if delta < threshold {
			// Apply floor: preserve at least 30% of initial activation
			// This prevents CrossFire from completely killing LOVE/FLOW
			result := make(map[string]float32)
			for i, name := range CHAMBER_NAMES {
				floor := initialActivations[i] * 0.3
				if activations[i] < floor {
					activations[i] = floor
				}
				result[name] = activations[i]
			}
			return result, iter + 1
		}
	}

	// Apply floor to final result as well
	result := make(map[string]float32)
	for i, name := range CHAMBER_NAMES {
		floor := initialActivations[i] * 0.3
		if activations[i] < floor {
			activations[i] = floor
		}
		result[name] = activations[i]
	}
	return result, maxIter
}

func (cfs *CrossFireSystem) ParamCount() int {
	total := 0
	for _, ch := range cfs.Chambers {
		total += ch.ParamCount()
	}
	return total
}

// ═══════════════════════════════════════════════════════════════════════════
// CLOUD — Async Pre-Processor
// "Something fires BEFORE meaning arrives"
// ═══════════════════════════════════════════════════════════════════════════

type CloudResponse struct {
	Primary            string
	Secondary          string
	Resonances         []float32
	ChamberActivations map[string]float32
	Iterations         int
	TemperatureBias    float32
}

type PingRequest struct {
	Text     string
	Response chan *CloudResponse
}

type Cloud struct {
	CrossFire *CrossFireSystem
	Observer  *MetaObserver
	Anchors   []string

	// Async infrastructure
	pingChan   chan *PingRequest
	stopChan   chan struct{}
	running    bool
	mu         sync.RWMutex
}

func NewCloud(seed int64) *Cloud {
	return &Cloud{
		CrossFire: NewCrossFireSystem(seed),
		Observer:  NewRandomObserver(seed + 100),
		Anchors:   GetAllAnchors(),
		pingChan:  make(chan *PingRequest, 100),
		stopChan:  make(chan struct{}),
		running:   false,
	}
}

// Start async processing loop on dedicated OS thread
func (c *Cloud) Start() {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return
	}
	c.running = true
	c.mu.Unlock()

	// Run processLoop on dedicated OS thread for CGO compatibility
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		c.processLoop()
	}()

	// Give the goroutine time to start
	runtime.Gosched()
	fmt.Println("[cloud] async preprocessor started (dedicated thread)")
}

// Stop async processing
func (c *Cloud) Stop() {
	c.mu.Lock()
	if !c.running {
		c.mu.Unlock()
		return
	}
	c.running = false
	c.mu.Unlock()

	close(c.stopChan)
	fmt.Println("[cloud] async preprocessor stopped")
}

// Main async processing loop
func (c *Cloud) processLoop() {
	for {
		select {
		case <-c.stopChan:
			return
		case req := <-c.pingChan:
			response := c.processSync(req.Text)
			req.Response <- response
		}
	}
}

// Async ping — returns immediately, result comes via channel
func (c *Cloud) PingAsync(text string) chan *CloudResponse {
	responseChan := make(chan *CloudResponse, 1)

	c.mu.RLock()
	running := c.running
	c.mu.RUnlock()

	if !running {
		// Fallback to sync if not started
		go func() {
			responseChan <- c.processSync(text)
		}()
		return responseChan
	}

	c.pingChan <- &PingRequest{
		Text:     text,
		Response: responseChan,
	}

	return responseChan
}

// Sync ping (blocking) — for compatibility
func (c *Cloud) Ping(text string) *CloudResponse {
	return c.processSync(text)
}

// Internal sync processing
func (c *Cloud) processSync(text string) *CloudResponse {
	// Debug: show what text we're processing
	if len(text) > 0 {
		fmt.Printf("[cloud] processing: %s\n", text[:min(len(text), 60)])
	}

	// 1. Resonance (weightless geometry)
	resonances := ComputeResonance(text)

	// 2. Find primary emotion
	primaryIdx := 0
	maxRes := resonances[0]
	for i, r := range resonances {
		if r > maxRes {
			maxRes = r
			primaryIdx = i
		}
	}

	// 3. Cross-fire stabilization (chambers run in parallel)
	activations, iterations := c.CrossFire.Stabilize(resonances, 10)

	// 4. Chamber activations as array
	chamberArray := make([]float32, 6)
	for i, name := range CHAMBER_NAMES {
		chamberArray[i] = activations[name]
	}

	// 5. Observer predicts secondary
	secondaryIdx := c.Observer.PredictSecondary(resonances, chamberArray, float32(iterations))

	// 6. Compute temperature bias for main model
	bias := float32(0.0)
	bias += activations["FEAR"] * 0.15
	bias += activations["RAGE"] * 0.10
	bias -= activations["LOVE"] * 0.10
	bias -= activations["VOID"] * 0.05
	if bias > 0.2 {
		bias = 0.2
	}
	if bias < -0.2 {
		bias = -0.2
	}

	return &CloudResponse{
		Primary:            c.Anchors[primaryIdx],
		Secondary:          c.Anchors[secondaryIdx],
		Resonances:         resonances,
		ChamberActivations: activations,
		Iterations:         iterations,
		TemperatureBias:    bias,
	}
}

func (c *Cloud) ParamCount() int {
	return c.CrossFire.ParamCount() + c.Observer.ParamCount()
}

// ═══════════════════════════════════════════════════════════════════════════
// WEIGHT LOADING
// ═══════════════════════════════════════════════════════════════════════════

func LoadChamberFromBin(path string) (*ChamberMLP, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := &ChamberMLP{}
	fmt.Printf("[cloud] loading chamber from %s\n", path)

	// Read number of arrays
	var numArrays uint32
	binary.Read(f, binary.LittleEndian, &numArrays)

	for i := uint32(0); i < numArrays; i++ {
		// Read name
		var nameLen uint32
		binary.Read(f, binary.LittleEndian, &nameLen)
		nameBytes := make([]byte, nameLen)
		f.Read(nameBytes)
		name := string(nameBytes)

		// Read shape
		var shapeLen uint32
		binary.Read(f, binary.LittleEndian, &shapeLen)
		shape := make([]uint32, shapeLen)
		for j := uint32(0); j < shapeLen; j++ {
			binary.Read(f, binary.LittleEndian, &shape[j])
		}

		// Read data
		var dataSize uint32
		binary.Read(f, binary.LittleEndian, &dataSize)
		data := make([]float32, dataSize)
		binary.Read(f, binary.LittleEndian, &data)

		// Assign to struct
		switch name {
		case "W1":
			m.W1 = data
		case "b1":
			m.B1 = data
		case "W2":
			m.W2 = data
		case "b2":
			m.B2 = data
		case "W3":
			m.W3 = data
		case "b3":
			m.B3 = data
		}
	}

	// Verify loaded sizes (expected: W1=6400, b1=64, W2=2048, b2=32, W3=32, b3=1)
	fmt.Printf("[cloud]   W1:%d b1:%d W2:%d b2:%d W3:%d b3:%d\n",
		len(m.W1), len(m.B1), len(m.W2), len(m.B2), len(m.W3), len(m.B3))

	return m, nil
}

func LoadCloud(weightsDir string) (*Cloud, error) {
	c := &Cloud{
		CrossFire: &CrossFireSystem{
			Chambers: make(map[string]*ChamberMLP),
		},
		Anchors:  GetAllAnchors(),
		pingChan: make(chan *PingRequest, 100),
		stopChan: make(chan struct{}),
		running:  false,
	}

	// Load chambers
	for i, name := range []string{"fear", "love", "rage", "void"} {
		path := filepath.Join(weightsDir, fmt.Sprintf("chamber_%s.bin", name))
		chamber, err := LoadChamberFromBin(path)
		if err != nil {
			fmt.Printf("[cloud] %s not found, using random init\n", path)
			chamber = NewRandomChamber(int64(i))
		}
		c.CrossFire.Chambers[strings.ToUpper(name)] = chamber
	}

	// Flow and Complex always random (not in original haze)
	c.CrossFire.Chambers["FLOW"] = NewRandomChamber(4)
	c.CrossFire.Chambers["COMPLEX"] = NewRandomChamber(5)

	// Load observer (TODO: implement actual loading from observer.bin)
	_ = filepath.Join(weightsDir, "observer.bin") // suppress unused warning
	c.Observer = NewRandomObserver(100)
	fmt.Printf("[cloud] observer loaded (random init for now)\n")

	fmt.Printf("[cloud] loaded from %s\n", weightsDir)
	fmt.Printf("[cloud] total params: %d\n", c.ParamCount())

	return c, nil
}

// ═══════════════════════════════════════════════════════════════════════════
// CGO EXPORTS — Interface for C
// ═══════════════════════════════════════════════════════════════════════════

var globalCloud *Cloud
var lastResponse *CloudResponse

//export cloud_init
func cloud_init(weightsDir *C.char) C.int {
	dir := C.GoString(weightsDir)
	var err error
	globalCloud, err = LoadCloud(dir)
	if err != nil {
		globalCloud = NewCloud(42)
		fmt.Printf("[cloud] using random init: %v\n", err)
	}
	// Start async preprocessor on dedicated thread
	globalCloud.Start()
	return 0
}

//export cloud_preprocess
func cloud_preprocess(text *C.char) C.int {
	if globalCloud == nil {
		globalCloud = NewCloud(42)
		globalCloud.Start()
	}

	goText := C.GoString(text)

	// Use async ping through the dedicated thread
	responseChan := globalCloud.PingAsync(goText)

	// Wait for result with timeout, yielding to allow async goroutine to run
	timeout := time.After(5 * time.Second)
	for {
		select {
		case result := <-responseChan:
			lastResponse = result
			return C.int(result.Iterations)
		case <-timeout:
			// Timeout — fallback to sync
			fmt.Println("[cloud] async timeout, using sync fallback")
			lastResponse = globalCloud.processSync(goText)
			return C.int(lastResponse.Iterations)
		default:
			// Yield to allow async goroutine to process
			runtime.Gosched()
		}
	}
}

//export cloud_get_temperature_bias
func cloud_get_temperature_bias() C.float {
	if lastResponse == nil {
		return 0.0
	}
	return C.float(lastResponse.TemperatureBias)
}

//export cloud_get_primary
func cloud_get_primary() *C.char {
	if lastResponse == nil {
		return C.CString("")
	}
	return C.CString(lastResponse.Primary)
}

//export cloud_get_secondary
func cloud_get_secondary() *C.char {
	if lastResponse == nil {
		return C.CString("")
	}
	return C.CString(lastResponse.Secondary)
}

//export cloud_get_chamber
func cloud_get_chamber(name *C.char) C.float {
	if lastResponse == nil {
		return 0.0
	}
	n := C.GoString(name)
	return C.float(lastResponse.ChamberActivations[n])
}

//export cloud_ping
func cloud_ping(text *C.char) *C.char {
	if globalCloud == nil {
		globalCloud = NewCloud(42)
	}

	response := globalCloud.Ping(C.GoString(text))
	lastResponse = response

	// Return primary|secondary|iterations|temp_bias|FEAR:0.5,LOVE:0.3,...
	result := fmt.Sprintf("%s|%s|%d|%.3f|",
		response.Primary, response.Secondary,
		response.Iterations, response.TemperatureBias)
	for _, name := range CHAMBER_NAMES {
		result += fmt.Sprintf("%s:%.3f,", name, response.ChamberActivations[name])
	}

	return C.CString(result)
}

//export cloud_stop
func cloud_stop() {
	if globalCloud != nil {
		globalCloud.Stop()
	}
}

//export cloud_free
func cloud_free() {
	if globalCloud != nil && globalCloud.running {
		globalCloud.Stop()
	}
	globalCloud = nil
	lastResponse = nil
}
