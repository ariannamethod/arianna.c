// high_ref_test.go — an independent Go implementation of High's structural
// formulas, used to verify the Julia brain without copying its output.

package main

import (
	"math"
	"strings"
	"unicode"
)

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
	for _, w := range words {
		wc[w]++
		tot++
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
	return ent, 0
}

func refValence(string) float64          { return 0 }
func refArousal(string) float64          { return 0 }
func refEmoAlign(string, string) float64 { return 1 }

func refPredSurprise(exp, act string) float64 {
	se := refSemDist(exp, act)
	en1, _ := refVecEntropy(exp)
	en2, _ := refVecEntropy(act)
	en := math.Abs(en1 - en2)
	return se*0.8 + en*0.2
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
	syl := make([]float64, len(words))
	for i, w := range words {
		syl[i] = float64(len([]rune(w)))
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
