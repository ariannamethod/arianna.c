// inner_world/high.go — High Mathematical Brain (ported from Julia/Python)
// ═══════════════════════════════════════════════════════════════════════════════
// המוח המתמטי של אריאנה
// Arianna's mathematical brain - fast computations
// ═══════════════════════════════════════════════════════════════════════════════
//
// Ported from high.py (KAIN/ADAM project)
// High provides fast mathematical operations:
// - Vectorized entropy with emotional weights
// - Semantic distance calculations
// - N-gram processing
// - Perplexity computation
//
// Philosophy: Mathematics is the high-level language of thought.
//
// ═══════════════════════════════════════════════════════════════════════════════

package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

// EmotionalWeights maps words to their emotional valence
// Positive values = positive emotion, Negative = negative
var EmotionalWeights = map[string]float32{
	// English - Positive
	"great": 0.8, "love": 0.9, "amazing": 0.7, "wonderful": 0.8, "excellent": 0.7,
	"beautiful": 0.8, "fantastic": 0.7, "awesome": 0.8, "perfect": 0.7, "brilliant": 0.8,
	"happy": 0.7, "joy": 0.8, "excited": 0.7, "delighted": 0.8, "pleased": 0.6,
	"good": 0.5, "nice": 0.4, "fine": 0.3, "okay": 0.1, "thanks": 0.4,
	"grateful": 0.7, "blessed": 0.6, "peaceful": 0.5, "calm": 0.4, "serene": 0.5,
	"hope": 0.6, "dream": 0.5, "inspire": 0.6, "create": 0.5, "grow": 0.4,

	// English - Negative
	"terrible": -0.8, "hate": -0.9, "awful": -0.7, "horrible": -0.8, "disgusting": -0.9,
	"sad": -0.6, "angry": -0.7, "frustrated": -0.6, "disappointed": -0.6, "upset": -0.6,
	"bad": -0.5, "wrong": -0.4, "fail": -0.6, "lose": -0.5, "hurt": -0.7,
	"pain": -0.8, "suffer": -0.8, "fear": -0.7, "anxiety": -0.6, "stress": -0.5,
	"alone": -0.6, "lonely": -0.7, "empty": -0.5, "nothing": -0.6, "worthless": -0.9,
	"stupid": -0.7, "ugly": -0.6, "weak": -0.5, "useless": -0.8, "pathetic": -0.8,

	// Russian - Positive
	"отлично":   0.8, "классно": 0.7, "супер": 0.8, "круто": 0.7, "прекрасно": 0.8,
	"здорово":   0.7, "замечательно": 0.8, "чудесно": 0.7, "великолепно": 0.8,
	"люблю":     0.9, "радость": 0.8, "счастье": 0.9, "мир": 0.5, "добро": 0.6,
	"красиво":   0.7, "хорошо": 0.5, "спасибо": 0.5, "благодарю": 0.6,

	// Russian - Negative
	"ужасно": -0.8, "плохо": -0.6, "грустно": -0.6, "злой": -0.7, "расстроен": -0.6,
	"больно": -0.8, "страшно": -0.7, "одиноко": -0.7, "пусто": -0.5, "ничто": -0.6,
	"ненавижу": -0.9, "страдаю": -0.8, "боюсь": -0.7, "тревога": -0.6,
	"глупый": -0.6, "слабый": -0.5, "никчёмный": -0.8, "жалкий": -0.7,

	// Hebrew - Positive
	"טוב": 0.5, "יפה": 0.7, "מדהים": 0.8, "נהדר": 0.7, "אהבה": 0.9,
	"שמחה": 0.8, "תקווה": 0.6, "שלום": 0.5, "ברכה": 0.6,

	// Hebrew - Negative
	"רע": -0.5, "נורא": -0.8, "עצוב": -0.6, "כועס": -0.7, "פחד": -0.7,
	"כאב": -0.8, "בודד": -0.7, "ריק": -0.5, "שנאה": -0.9,

	// Additional trauma triggers (unique ones not already above)
	"die":       -0.9, "kill": -0.9, "failure": -0.8, "loser": -0.8,
	"reject":    -0.7, "abandon": -0.8, "betray": -0.8, "forget": -0.5,
	"ignore":    -0.6, "invisible": -0.7, "broken": -0.7, "damaged": -0.7,
	"ruined":    -0.7, "trapped": -0.7, "hopeless": -0.8, "lost": -0.5,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIGH MATH ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

// HighMathEngine provides fast mathematical computations
type HighMathEngine struct {
	ngramCache   map[string][]string
	entropyCache map[string]float32
	mu           sync.RWMutex
}

// NewHighMathEngine creates the math engine
func NewHighMathEngine() *HighMathEngine {
	return &HighMathEngine{
		ngramCache:   make(map[string][]string),
		entropyCache: make(map[string]float32),
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENTROPY CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// VectorizedEntropy computes entropy with emotional modulation
func (h *HighMathEngine) VectorizedEntropy(texts []string) (entropy float32, emotionalScore float32) {
	if len(texts) == 0 {
		return 0, 0
	}

	wordCounts := make(map[string]int)
	totalWords := 0
	emotionalSum := float32(0)

	for _, text := range texts {
		words := tokenize(text)
		for _, word := range words {
			wordCounts[word]++
			totalWords++

			// Accumulate emotional weight
			if weight, exists := EmotionalWeights[word]; exists {
				emotionalSum += weight
			}
		}
	}

	if totalWords == 0 {
		return 0, 0
	}

	// Compute Shannon entropy
	entropy = float32(0)
	for _, count := range wordCounts {
		p := float32(count) / float32(totalWords)
		if p > 0 {
			entropy -= p * float32(math.Log2(float64(p)))
		}
	}

	// Emotional score normalized by word count
	emotionalScore = emotionalSum / float32(totalWords)

	// Modulate entropy by emotional intensity
	// High emotion (positive or negative) slightly increases perceived entropy
	emotionalMod := 1.0 + float32(math.Abs(float64(emotionalScore)))*0.2
	entropy *= emotionalMod

	return entropy, emotionalScore
}

// CharEntropy computes character-level entropy
func (h *HighMathEngine) CharEntropy(text string) float32 {
	if len(text) == 0 {
		return 0
	}

	charCounts := make(map[rune]int)
	total := 0

	for _, c := range text {
		charCounts[c]++
		total++
	}

	entropy := float32(0)
	for _, count := range charCounts {
		p := float32(count) / float32(total)
		if p > 0 {
			entropy -= p * float32(math.Log2(float64(p)))
		}
	}

	return entropy
}

// ═══════════════════════════════════════════════════════════════════════════════
// N-GRAM PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

// ExtractNgrams extracts n-grams from text
func (h *HighMathEngine) ExtractNgrams(text string, n int) []string {
	h.mu.RLock()
	cacheKey := text + string(rune(n))
	if cached, exists := h.ngramCache[cacheKey]; exists {
		h.mu.RUnlock()
		return cached
	}
	h.mu.RUnlock()

	words := tokenize(text)
	if len(words) < n {
		return nil
	}

	ngrams := make([]string, 0, len(words)-n+1)
	for i := 0; i <= len(words)-n; i++ {
		ngram := strings.Join(words[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}

	// Cache result
	h.mu.Lock()
	h.ngramCache[cacheKey] = ngrams
	h.mu.Unlock()

	return ngrams
}

// NgramOverlap computes overlap between two texts' n-grams
func (h *HighMathEngine) NgramOverlap(text1, text2 string, n int) float32 {
	ngrams1 := h.ExtractNgrams(text1, n)
	ngrams2 := h.ExtractNgrams(text2, n)

	if len(ngrams1) == 0 || len(ngrams2) == 0 {
		return 0
	}

	set1 := make(map[string]bool)
	for _, ng := range ngrams1 {
		set1[ng] = true
	}

	overlap := 0
	for _, ng := range ngrams2 {
		if set1[ng] {
			overlap++
		}
	}

	// Jaccard similarity
	union := len(ngrams1) + len(ngrams2) - overlap
	if union == 0 {
		return 0
	}

	return float32(overlap) / float32(union)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC DISTANCE
// ═══════════════════════════════════════════════════════════════════════════════

// SemanticDistance computes distance between two texts
// Uses bag-of-words cosine similarity + emotional alignment
func (h *HighMathEngine) SemanticDistance(text1, text2 string) float32 {
	words1 := tokenize(text1)
	words2 := tokenize(text2)

	if len(words1) == 0 || len(words2) == 0 {
		return 1.0 // Maximum distance
	}

	// Build vocabulary
	vocab := make(map[string]int)
	idx := 0
	for _, w := range words1 {
		if _, exists := vocab[w]; !exists {
			vocab[w] = idx
			idx++
		}
	}
	for _, w := range words2 {
		if _, exists := vocab[w]; !exists {
			vocab[w] = idx
			idx++
		}
	}

	// Build vectors
	vec1 := make([]float32, len(vocab))
	vec2 := make([]float32, len(vocab))

	for _, w := range words1 {
		vec1[vocab[w]]++
	}
	for _, w := range words2 {
		vec2[vocab[w]]++
	}

	// Cosine similarity
	dot := float32(0)
	norm1 := float32(0)
	norm2 := float32(0)

	for i := range vec1 {
		dot += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 1.0
	}

	similarity := dot / (float32(math.Sqrt(float64(norm1))) * float32(math.Sqrt(float64(norm2))))

	// Convert to distance
	return 1.0 - similarity
}

// EmotionalAlignment computes how emotionally aligned two texts are
func (h *HighMathEngine) EmotionalAlignment(text1, text2 string) float32 {
	_, emotion1 := h.VectorizedEntropy([]string{text1})
	_, emotion2 := h.VectorizedEntropy([]string{text2})

	// Aligned if same sign and similar magnitude
	if (emotion1 >= 0) == (emotion2 >= 0) {
		// Same sign - compute similarity
		diff := float32(math.Abs(float64(emotion1 - emotion2)))
		maxMag := float32(math.Max(math.Abs(float64(emotion1)), math.Abs(float64(emotion2))))
		if maxMag == 0 {
			return 1.0 // Both neutral
		}
		return 1.0 - diff/maxMag
	}

	// Opposite signs - negative alignment
	return -float32(math.Abs(float64(emotion1-emotion2))) / 2
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERPLEXITY
// ═══════════════════════════════════════════════════════════════════════════════

// Perplexity computes perplexity of text based on character transitions
func (h *HighMathEngine) Perplexity(text string) float32 {
	if len(text) < 2 {
		return 1.0
	}

	// Build bigram counts
	bigramCounts := make(map[string]int)
	unigramCounts := make(map[rune]int)

	runes := []rune(text)
	for i := 0; i < len(runes)-1; i++ {
		bigram := string(runes[i : i+2])
		bigramCounts[bigram]++
		unigramCounts[runes[i]]++
	}
	unigramCounts[runes[len(runes)-1]]++

	// Compute log probability
	logProb := float64(0)
	count := 0

	for i := 0; i < len(runes)-1; i++ {
		bigram := string(runes[i : i+2])
		bigramCount := bigramCounts[bigram]
		unigramCount := unigramCounts[runes[i]]

		if unigramCount > 0 && bigramCount > 0 {
			p := float64(bigramCount) / float64(unigramCount)
			logProb += math.Log2(p)
			count++
		}
	}

	if count == 0 {
		return 1.0
	}

	avgLogProb := logProb / float64(count)
	perplexity := math.Pow(2, -avgLogProb)

	return float32(perplexity)
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

// AnalyzeEmotion returns detailed emotional analysis of text
type EmotionalAnalysis struct {
	OverallScore   float32
	PositiveWords  []string
	NegativeWords  []string
	DominantTone   string
	Intensity      float32
	Arousal        float32
	Valence        float32
}

// AnalyzeEmotion performs detailed emotional analysis
func (h *HighMathEngine) AnalyzeEmotion(text string) EmotionalAnalysis {
	words := tokenize(text)

	analysis := EmotionalAnalysis{
		PositiveWords: make([]string, 0),
		NegativeWords: make([]string, 0),
	}

	totalScore := float32(0)
	wordCount := 0

	for _, word := range words {
		if weight, exists := EmotionalWeights[word]; exists {
			totalScore += weight
			wordCount++

			if weight > 0 {
				analysis.PositiveWords = append(analysis.PositiveWords, word)
			} else if weight < 0 {
				analysis.NegativeWords = append(analysis.NegativeWords, word)
			}
		}
	}

	if wordCount > 0 {
		analysis.OverallScore = totalScore / float32(wordCount)
	}

	// Determine dominant tone
	if analysis.OverallScore > 0.3 {
		analysis.DominantTone = "positive"
	} else if analysis.OverallScore < -0.3 {
		analysis.DominantTone = "negative"
	} else {
		analysis.DominantTone = "neutral"
	}

	// Intensity is absolute value of score
	analysis.Intensity = float32(math.Abs(float64(analysis.OverallScore)))

	// Map to valence-arousal space
	analysis.Valence = analysis.OverallScore
	// Arousal increases with intensity and word count
	analysis.Arousal = clamp(analysis.Intensity+float32(wordCount)*0.01, 0, 1)

	return analysis
}

// ═══════════════════════════════════════════════════════════════════════════════
// MATH FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Sigmoid activation
func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// Tanh activation
func Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// ReLU activation
func ReLU(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

// Softmax over a slice
func Softmax(x []float32) []float32 {
	if len(x) == 0 {
		return nil
	}

	// Find max for numerical stability
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}

	// Compute exp and sum
	result := make([]float32, len(x))
	sum := float32(0)
	for i, v := range x {
		result[i] = float32(math.Exp(float64(v - max)))
		sum += result[i]
	}

	// Normalize
	for i := range result {
		result[i] /= sum
	}

	return result
}

// TopK returns indices of top k values
func TopK(x []float32, k int) []int {
	if k > len(x) {
		k = len(x)
	}

	type indexedValue struct {
		index int
		value float32
	}

	indexed := make([]indexedValue, len(x))
	for i, v := range x {
		indexed[i] = indexedValue{i, v}
	}

	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].value > indexed[j].value
	})

	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = indexed[i].index
	}

	return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

// tokenize splits text into lowercase words
func tokenize(text string) []string {
	text = strings.ToLower(text)

	var words []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else if current.Len() > 0 {
			words = append(words, current.String())
			current.Reset()
		}
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBAL INSTANCE
// ═══════════════════════════════════════════════════════════════════════════════

var (
	globalHigh *HighMathEngine
	highMu     sync.Mutex
)

// GetHighMathEngine returns the global math engine
func GetHighMathEngine() *HighMathEngine {
	highMu.Lock()
	defer highMu.Unlock()

	if globalHigh == nil {
		globalHigh = NewHighMathEngine()
	}
	return globalHigh
}

// ═══════════════════════════════════════════════════════════════════════════════
// CGO EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

//export high_entropy
func high_entropy(text *C.char) C.float {
	goText := C.GoString(text)
	entropy, _ := GetHighMathEngine().VectorizedEntropy([]string{goText})
	return C.float(entropy)
}

//export high_emotional_score
func high_emotional_score(text *C.char) C.float {
	goText := C.GoString(text)
	_, emotional := GetHighMathEngine().VectorizedEntropy([]string{goText})
	return C.float(emotional)
}

//export high_perplexity
func high_perplexity(text *C.char) C.float {
	goText := C.GoString(text)
	return C.float(GetHighMathEngine().Perplexity(goText))
}

//export high_semantic_distance
func high_semantic_distance(text1 *C.char, text2 *C.char) C.float {
	goText1 := C.GoString(text1)
	goText2 := C.GoString(text2)
	return C.float(GetHighMathEngine().SemanticDistance(goText1, goText2))
}

//export high_emotional_alignment
func high_emotional_alignment(text1 *C.char, text2 *C.char) C.float {
	goText1 := C.GoString(text1)
	goText2 := C.GoString(text2)
	return C.float(GetHighMathEngine().EmotionalAlignment(goText1, goText2))
}

//export high_ngram_overlap
func high_ngram_overlap(text1 *C.char, text2 *C.char, n C.int) C.float {
	goText1 := C.GoString(text1)
	goText2 := C.GoString(text2)
	return C.float(GetHighMathEngine().NgramOverlap(goText1, goText2, int(n)))
}

//export high_analyze_valence
func high_analyze_valence(text *C.char) C.float {
	goText := C.GoString(text)
	analysis := GetHighMathEngine().AnalyzeEmotion(goText)
	return C.float(analysis.Valence)
}

//export high_analyze_arousal
func high_analyze_arousal(text *C.char) C.float {
	goText := C.GoString(text)
	analysis := GetHighMathEngine().AnalyzeEmotion(goText)
	return C.float(analysis.Arousal)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ODE-BASED EMOTIONAL DYNAMICS (Friston Free Energy Principle)
// "Emotions evolve through differential equations, not discrete jumps"
// ═══════════════════════════════════════════════════════════════════════════════

// EmotionalState represents the current emotional configuration
type EmotionalState struct {
	Valence    float32 // [-1, 1] negative to positive
	Arousal    float32 // [0, 1] calm to excited
	Entropy    float32 // [0, inf] uncertainty/chaos
	Prediction float32 // Expected valence (for surprise calculation)
}

// EmotionalDriftParams controls the ODE dynamics
type EmotionalDriftParams struct {
	DecayRate       float32 // Return to baseline speed (τ)
	SurpriseGain    float32 // How much surprise affects state
	EntropyWeight   float32 // Entropy influence on arousal
	BaselineValence float32 // Resting valence
	BaselineArousal float32 // Resting arousal
}

// DefaultDriftParams returns sensible defaults
func DefaultDriftParams() EmotionalDriftParams {
	return EmotionalDriftParams{
		DecayRate:       0.1,  // Slow return to baseline
		SurpriseGain:    0.3,  // Moderate surprise sensitivity
		EntropyWeight:   0.2,  // Entropy slightly increases arousal
		BaselineValence: 0.1,  // Slightly positive baseline
		BaselineArousal: 0.3,  // Calm but alert baseline
	}
}

// EmotionalDrift computes the next emotional state using ODE
// dV/dt = -τ(V - V₀) + surprise * gain
// dA/dt = -τ(A - A₀) + |surprise| * gain + entropy * weight
func (h *HighMathEngine) EmotionalDrift(
	current EmotionalState,
	input string,
	dt float32,
	params EmotionalDriftParams,
) EmotionalState {
	// Analyze input
	analysis := h.AnalyzeEmotion(input)
	inputEntropy, _ := h.VectorizedEntropy([]string{input})

	// Compute surprise (prediction error) - Free Energy Principle
	surprise := analysis.Valence - current.Prediction

	// ODE integration (Euler method)
	// dV/dt = decay toward baseline + surprise influence
	dValence := -params.DecayRate*(current.Valence-params.BaselineValence) +
		surprise*params.SurpriseGain

	// dA/dt = decay toward baseline + |surprise| + entropy
	dArousal := -params.DecayRate*(current.Arousal-params.BaselineArousal) +
		float32(math.Abs(float64(surprise)))*params.SurpriseGain +
		inputEntropy*params.EntropyWeight

	// Update state
	next := EmotionalState{
		Valence:    clamp(current.Valence+dValence*dt, -1, 1),
		Arousal:    clamp(current.Arousal+dArousal*dt, 0, 1),
		Entropy:    inputEntropy,
		Prediction: current.Valence + dValence*dt*0.5, // Predict next valence
	}

	return next
}

// PredictiveSurprise computes Free Energy (prediction error)
// Lower = better prediction, Higher = more surprise
func (h *HighMathEngine) PredictiveSurprise(expected, actual string) float32 {
	// Semantic distance as prediction error
	semanticError := h.SemanticDistance(expected, actual)

	// Emotional misalignment
	emotionalError := 1.0 - h.EmotionalAlignment(expected, actual)
	if emotionalError < 0 {
		emotionalError = -emotionalError
	}

	// Entropy difference
	entropyExp, _ := h.VectorizedEntropy([]string{expected})
	entropyAct, _ := h.VectorizedEntropy([]string{actual})
	entropyError := float32(math.Abs(float64(entropyExp - entropyAct)))

	// Combined surprise (Free Energy proxy)
	freeEnergy := semanticError*0.4 + emotionalError*0.4 + entropyError*0.2

	return freeEnergy
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESONANCE COUPLING
// "Internal states couple with external rhythms"
// ═══════════════════════════════════════════════════════════════════════════════

// SchumannResonance base frequency
const SchumannBaseHz = 7.83

// ResonanceCoupling computes coupling strength between internal state and text
// Based on how well the text "resonates" with current emotional configuration
func (h *HighMathEngine) ResonanceCoupling(
	internal EmotionalState,
	external string,
	schumannMod float32, // Schumann resonance modulation [0.9, 1.1]
) float32 {
	// Analyze external text
	analysis := h.AnalyzeEmotion(external)

	// Valence alignment
	valenceAlign := 1.0 - float32(math.Abs(float64(internal.Valence-analysis.Valence)))/2.0

	// Arousal alignment
	arousalAlign := 1.0 - float32(math.Abs(float64(internal.Arousal-analysis.Arousal)))

	// Entropy coupling (similar entropy = better coupling)
	extEntropy, _ := h.VectorizedEntropy([]string{external})
	entropyCoupling := 1.0 - float32(math.Abs(float64(internal.Entropy-extEntropy)))/5.0
	if entropyCoupling < 0 {
		entropyCoupling = 0
	}

	// Base coupling
	coupling := valenceAlign*0.4 + arousalAlign*0.3 + entropyCoupling*0.3

	// Schumann modulation (coherence with Earth's resonance)
	// schumannMod near 1.0 = coherent, far from 1.0 = disrupted
	schumannCoherence := 1.0 - float32(math.Abs(float64(schumannMod-1.0)))*5.0
	if schumannCoherence < 0.5 {
		schumannCoherence = 0.5
	}

	return coupling * schumannCoherence
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEXT RHYTHM ANALYSIS
// "Language has rhythm, rhythm has meaning"
// ═══════════════════════════════════════════════════════════════════════════════

// TextRhythm analyzes syllabic rhythm of text
// Returns: avgSyllablesPerWord, rhythmVariance, pauseDensity
func (h *HighMathEngine) TextRhythm(text string) (avgSyllables float32, variance float32, pauses float32) {
	words := tokenize(text)
	if len(words) == 0 {
		return 0, 0, 0
	}

	// Estimate syllables per word (simple: count vowels)
	syllableCounts := make([]float32, len(words))
	vowels := "aeiouyаеёиоуыэюяאֵֶֻוִ"

	for i, word := range words {
		count := float32(0)
		for _, r := range word {
			if strings.ContainsRune(vowels, r) {
				count++
			}
		}
		if count < 1 {
			count = 1 // Minimum 1 syllable
		}
		syllableCounts[i] = count
	}

	// Average syllables
	sum := float32(0)
	for _, c := range syllableCounts {
		sum += c
	}
	avgSyllables = sum / float32(len(words))

	// Variance (rhythm regularity)
	varSum := float32(0)
	for _, c := range syllableCounts {
		diff := c - avgSyllables
		varSum += diff * diff
	}
	variance = varSum / float32(len(words))

	// Pause density (punctuation / words)
	pauseCount := float32(strings.Count(text, ",") + strings.Count(text, ".") +
		strings.Count(text, ";") + strings.Count(text, "—") +
		strings.Count(text, "..."))
	pauses = pauseCount / float32(len(words))

	return avgSyllables, variance, pauses
}

// ═══════════════════════════════════════════════════════════════════════════════
// CGO EXPORTS - NEW FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

//export high_predictive_surprise
func high_predictive_surprise(expected *C.char, actual *C.char) C.float {
	goExpected := C.GoString(expected)
	goActual := C.GoString(actual)
	return C.float(GetHighMathEngine().PredictiveSurprise(goExpected, goActual))
}

//export high_resonance_coupling
func high_resonance_coupling(valence C.float, arousal C.float, entropy C.float, text *C.char, schumann C.float) C.float {
	state := EmotionalState{
		Valence: float32(valence),
		Arousal: float32(arousal),
		Entropy: float32(entropy),
	}
	goText := C.GoString(text)
	return C.float(GetHighMathEngine().ResonanceCoupling(state, goText, float32(schumann)))
}

//export high_text_rhythm_avg
func high_text_rhythm_avg(text *C.char) C.float {
	goText := C.GoString(text)
	avg, _, _ := GetHighMathEngine().TextRhythm(goText)
	return C.float(avg)
}

//export high_text_rhythm_variance
func high_text_rhythm_variance(text *C.char) C.float {
	goText := C.GoString(text)
	_, variance, _ := GetHighMathEngine().TextRhythm(goText)
	return C.float(variance)
}
