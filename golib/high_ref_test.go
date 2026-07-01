// high_ref_test.go — an INDEPENDENT Go reimplementation of the legacy HighMathEngine
// formulas, used to verify high.jl (real Julia) is faithful — not a self-confirming snapshot.
// EmotionalWeights is copied verbatim from the legacy map; the math mirrors legacy semantics
// in float64 (high.jl computes float64).

package main

import (
	"math"
	"strings"
	"unicode"
)

var emotionalWeightsRef = map[string]float64{
	"great": 0.8, "love": 0.9, "amazing": 0.7, "wonderful": 0.8, "excellent": 0.7,
	"beautiful": 0.8, "fantastic": 0.7, "awesome": 0.8, "perfect": 0.7, "brilliant": 0.8,
	"happy": 0.7, "joy": 0.8, "excited": 0.7, "delighted": 0.8, "pleased": 0.6,
	"good": 0.5, "nice": 0.4, "fine": 0.3, "okay": 0.1, "thanks": 0.4,
	"grateful": 0.7, "blessed": 0.6, "peaceful": 0.5, "calm": 0.4, "serene": 0.5,
	"hope": 0.6, "dream": 0.5, "inspire": 0.6, "create": 0.5, "grow": 0.4,
	"terrible": -0.8, "hate": -0.9, "awful": -0.7, "horrible": -0.8, "disgusting": -0.9,
	"sad": -0.6, "angry": -0.7, "frustrated": -0.6, "disappointed": -0.6, "upset": -0.6,
	"bad": -0.5, "wrong": -0.4, "fail": -0.6, "lose": -0.5, "hurt": -0.7,
	"pain": -0.8, "suffer": -0.8, "fear": -0.7, "anxiety": -0.6, "stress": -0.5,
	"alone": -0.6, "lonely": -0.7, "empty": -0.5, "nothing": -0.6, "worthless": -0.9,
	"stupid": -0.7, "ugly": -0.6, "weak": -0.5, "useless": -0.8, "pathetic": -0.8,
	"отлично": 0.8, "классно": 0.7, "супер": 0.8, "круто": 0.7, "прекрасно": 0.8,
	"здорово": 0.7, "замечательно": 0.8, "чудесно": 0.7, "великолепно": 0.8,
	"люблю": 0.9, "радость": 0.8, "счастье": 0.9, "мир": 0.5, "добро": 0.6,
	"красиво": 0.7, "хорошо": 0.5, "спасибо": 0.5, "благодарю": 0.6,
	"ужасно": -0.8, "плохо": -0.6, "грустно": -0.6, "злой": -0.7, "расстроен": -0.6,
	"больно": -0.8, "страшно": -0.7, "одиноко": -0.7, "пусто": -0.5, "ничто": -0.6,
	"ненавижу": -0.9, "страдаю": -0.8, "боюсь": -0.7, "тревога": -0.6,
	"глупый": -0.6, "слабый": -0.5, "никчёмный": -0.8, "жалкий": -0.7,
	"טוב": 0.5, "יפה": 0.7, "מדהים": 0.8, "נהדר": 0.7, "אהבה": 0.9,
	"שמחה": 0.8, "תקווה": 0.6, "שלום": 0.5, "ברכה": 0.6,
	"רע": -0.5, "נורא": -0.8, "עצוב": -0.6, "כועס": -0.7, "פחד": -0.7,
	"כאב": -0.8, "בודד": -0.7, "ריק": -0.5, "שנאה": -0.9,
	"die": -0.9, "kill": -0.9, "failure": -0.8, "loser": -0.8,
	"reject": -0.7, "abandon": -0.8, "betray": -0.8, "forget": -0.5,
	"ignore": -0.6, "invisible": -0.7, "broken": -0.7, "damaged": -0.7,
	"ruined": -0.7, "trapped": -0.7, "hopeless": -0.8, "lost": -0.5,
}

func refTokenize(s string) []string {
	s = strings.ToLower(s)
	var w []string
	var cur []rune
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur = append(cur, r)
		} else if len(cur) > 0 {
			w = append(w, string(cur))
			cur = nil
		}
	}
	if len(cur) > 0 {
		w = append(w, string(cur))
	}
	return w
}

func refCharEntropy(s string) float64 {
	if s == "" {
		return 0
	}
	cnt := map[rune]int{}
	tot := 0
	for _, c := range s {
		cnt[c]++
		tot++
	}
	h := 0.0
	for _, c := range cnt {
		p := float64(c) / float64(tot)
		if p > 0 {
			h -= p * math.Log2(p)
		}
	}
	return h
}

func refPerplexity(s string) float64 {
	r := []rune(s)
	n := len(r)
	if n < 2 {
		return 1.0
	}
	bg := map[[2]rune]int{}
	uni := map[rune]int{}
	for i := 0; i < n-1; i++ {
		bg[[2]rune{r[i], r[i+1]}]++
		uni[r[i]]++
	}
	uni[r[n-1]]++
	lp := 0.0
	c := 0
	for i := 0; i < n-1; i++ {
		b := bg[[2]rune{r[i], r[i+1]}]
		u := uni[r[i]]
		if u > 0 && b > 0 {
			lp += math.Log2(float64(b) / float64(u))
			c++
		}
	}
	if c == 0 {
		return 1.0
	}
	return math.Pow(2, -(lp / float64(c)))
}

func refNgrams(s string, n int) []string {
	w := refTokenize(s)
	if len(w) < n {
		return nil
	}
	var g []string
	for i := 0; i <= len(w)-n; i++ {
		g = append(g, strings.Join(w[i:i+n], " "))
	}
	return g
}

func refNgramOverlap(a, b string, n int) float64 {
	g1 := refNgrams(a, n)
	g2 := refNgrams(b, n)
	if len(g1) == 0 || len(g2) == 0 {
		return 0
	}
	s1 := map[string]bool{}
	for _, g := range g1 {
		s1[g] = true
	}
	ov := 0
	for _, g := range g2 {
		if s1[g] {
			ov++
		}
	}
	un := len(g1) + len(g2) - ov
	if un == 0 {
		return 0
	}
	return float64(ov) / float64(un)
}

func refSemDist(a, b string) float64 {
	w1 := refTokenize(a)
	w2 := refTokenize(b)
	if len(w1) == 0 || len(w2) == 0 {
		return 1.0
	}
	vocab := map[string]int{}
	idx := 0
	for _, w := range w1 {
		if _, ok := vocab[w]; !ok {
			vocab[w] = idx
			idx++
		}
	}
	for _, w := range w2 {
		if _, ok := vocab[w]; !ok {
			vocab[w] = idx
			idx++
		}
	}
	v1 := make([]float64, idx)
	v2 := make([]float64, idx)
	for _, w := range w1 {
		v1[vocab[w]]++
	}
	for _, w := range w2 {
		v2[vocab[w]]++
	}
	var d, n1, n2 float64
	for i := range v1 {
		d += v1[i] * v2[i]
		n1 += v1[i] * v1[i]
		n2 += v2[i] * v2[i]
	}
	if n1 == 0 || n2 == 0 {
		return 1.0
	}
	return 1.0 - d/(math.Sqrt(n1)*math.Sqrt(n2))
}

func refVecEntropy(s string) (float64, float64) {
	words := refTokenize(s)
	if len(words) == 0 {
		return 0, 0
	}
	wc := map[string]int{}
	tot := 0
	es := 0.0
	for _, w := range words {
		wc[w]++
		tot++
		if v, ok := emotionalWeightsRef[w]; ok {
			es += v
		}
	}
	if tot == 0 {
		return 0, 0
	}
	ent := 0.0
	for _, c := range wc {
		p := float64(c) / float64(tot)
		if p > 0 {
			ent -= p * math.Log2(p)
		}
	}
	esc := es / float64(tot)
	ent *= 1.0 + math.Abs(esc)*0.2
	return ent, esc
}

func refValence(s string) float64 {
	words := refTokenize(s)
	tot := 0.0
	c := 0
	for _, w := range words {
		if v, ok := emotionalWeightsRef[w]; ok {
			tot += v
			c++
		}
	}
	if c == 0 {
		return 0
	}
	return tot / float64(c)
}

func refArousal(s string) float64 {
	words := refTokenize(s)
	tot := 0.0
	c := 0
	for _, w := range words {
		if v, ok := emotionalWeightsRef[w]; ok {
			tot += v
			c++
		}
	}
	val := 0.0
	if c > 0 {
		val = tot / float64(c)
	}
	ar := math.Abs(val) + float64(c)*0.01
	if ar < 0 {
		ar = 0
	}
	if ar > 1 {
		ar = 1
	}
	return ar
}

func refEmoAlign(a, b string) float64 {
	_, e1 := refVecEntropy(a)
	_, e2 := refVecEntropy(b)
	if (e1 >= 0) == (e2 >= 0) {
		diff := math.Abs(e1 - e2)
		mx := math.Max(math.Abs(e1), math.Abs(e2))
		if mx == 0 {
			return 1.0
		}
		return 1.0 - diff/mx
	}
	return -math.Abs(e1-e2) / 2
}

func refPredSurprise(exp, act string) float64 {
	se := refSemDist(exp, act)
	ee := math.Abs(1.0 - refEmoAlign(exp, act))
	en1, _ := refVecEntropy(exp)
	en2, _ := refVecEntropy(act)
	en := math.Abs(en1 - en2)
	return se*0.4 + ee*0.4 + en*0.2
}

func refResonance(val, aro, ent float64, ext string, sch float64) float64 {
	av := refValence(ext)
	aa := refArousal(ext)
	valign := 1.0 - math.Abs(val-av)/2.0
	aalign := 1.0 - math.Abs(aro-aa)
	ee, _ := refVecEntropy(ext)
	ecoup := 1.0 - math.Abs(ent-ee)/5.0
	if ecoup < 0 {
		ecoup = 0
	}
	coup := valign*0.4 + aalign*0.3 + ecoup*0.3
	s := 1.0 - math.Abs(sch-1.0)*5.0
	if s < 0.5 {
		s = 0.5
	}
	return coup * s
}

func refRhythm(s string) (float64, float64, float64) {
	words := refTokenize(s)
	if len(words) == 0 {
		return 0, 0, 0
	}
	vowels := "aeiouyаеёиоуыэюяאֵֶֻוִ"
	syl := make([]float64, len(words))
	for i, w := range words {
		cc := 0.0
		for _, r := range w {
			if strings.ContainsRune(vowels, r) {
				cc++
			}
		}
		if cc < 1 {
			cc = 1
		}
		syl[i] = cc
	}
	sum := 0.0
	for _, c := range syl {
		sum += c
	}
	avg := sum / float64(len(words))
	vs := 0.0
	for _, c := range syl {
		d := c - avg
		vs += d * d
	}
	variance := vs / float64(len(words))
	pc := float64(strings.Count(s, ",") + strings.Count(s, ".") + strings.Count(s, ";") + strings.Count(s, "—") + strings.Count(s, "..."))
	pauses := pc / float64(len(words))
	return avg, variance, pauses
}

func refSigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
func refReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func refTanh(x float64) float64 { return math.Tanh(x) }
