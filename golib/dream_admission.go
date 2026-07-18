package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode"
)

const (
	dreamAdmissionLive   = "live"
	dreamAdmissionShadow = "shadow"
)

// dreamCandidate is the typed boundary between observing a subconscious trace
// and letting it mutate Arianna.c state. Default live mode preserves existing
// behavior; AM_DREAM_ADMISSION=shadow records the candidate and rejects mutation.
type dreamCandidate struct {
	Schema    string    `json:"schema"`
	RunID     string    `json:"run_id"`
	Mode      string    `json:"mode"`
	Source    string    `json:"source"`
	Trigger   string    `json:"trigger"`
	Seed      string    `json:"seed"`
	Fragment  string    `json:"fragment,omitempty"`
	Text      string    `json:"text"`
	Kind      string    `json:"kind"`
	Cells     int       `json:"cells"`
	Questions int       `json:"questions"`
	Accepted  bool      `json:"accepted"`
	Reason    string    `json:"reason"`
	Created   time.Time `json:"created"`

	Counterfactual *dreamCounterfactual `json:"counterfactual,omitempty"`
}

type dreamCounterfactual struct {
	Schema        string             `json:"schema"`
	Target        string             `json:"target"`
	PreStateHash  string             `json:"pre_state_hash"`
	PostStateHash string             `json:"post_state_hash"`
	Delta         dreamSnapshotDelta `json:"delta"`
	Analysis      TextAnalysis       `json:"analysis"`
	Text          dreamTextMetrics   `json:"text"`
}

type dreamSnapshotDelta struct {
	Arousal          float32 `json:"arousal"`
	Valence          float32 `json:"valence"`
	Entropy          float32 `json:"entropy"`
	Coherence        float32 `json:"coherence"`
	TraumaLevel      float32 `json:"trauma_level"`
	LoopCount        int     `json:"loop_count"`
	AbstractionDepth int     `json:"abstraction_depth"`
	SelfRefCount     int     `json:"self_ref_count"`
	MemoryPressure   float32 `json:"memory_pressure"`
	FocusStrength    float32 `json:"focus_strength"`
	WanderPull       float32 `json:"wander_pull"`
	ProphecyDebt     float32 `json:"prophecy_debt"`
}

type dreamTextMetrics struct {
	Bytes          int      `json:"bytes"`
	Runes          int      `json:"runes"`
	Words          int      `json:"words"`
	LatinRunes     int      `json:"latin_runes"`
	CyrillicRunes  int      `json:"cyrillic_runes"`
	LanguageHint   string   `json:"language_hint"`
	RecipientTerms []string `json:"recipient_terms,omitempty"`
}

func dreamAdmissionMode() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION"))) {
	case "", dreamAdmissionLive, "admit":
		return dreamAdmissionLive
	case dreamAdmissionShadow, "observe":
		return dreamAdmissionShadow
	default:
		// Unknown mode fails closed into observation. A misspelled admission knob
		// should never widen self-feedback by accident.
		return dreamAdmissionShadow
	}
}

func newDreamCandidate(source, trigger, seed, fragment, text string, cells []chorusCell) dreamCandidate {
	voices, questions := chorusCounts(cells)
	kind := source
	if kind == "" {
		kind = "unknown"
	}
	h := sha256.Sum256([]byte(source + "\x00" + trigger + "\x00" + seed + "\x00" + fragment + "\x00" + text))
	return dreamCandidate{
		Schema:    "arianna.dream_candidate.v1",
		RunID:     hex.EncodeToString(h[:8]),
		Source:    source,
		Trigger:   trigger,
		Seed:      seed,
		Fragment:  fragment,
		Text:      strings.TrimSpace(text),
		Kind:      kind,
		Cells:     voices,
		Questions: questions,
		Created:   time.Now().UTC(),
	}
}

func decideDreamCandidate(c dreamCandidate) dreamCandidate {
	c.Mode = dreamAdmissionMode()
	if strings.TrimSpace(c.Text) == "" {
		c.Accepted = false
		c.Reason = "empty dream"
		return c
	}
	if c.Mode == dreamAdmissionShadow {
		c.Accepted = false
		c.Reason = "shadow mode"
		return c
	}
	c.Accepted = true
	c.Reason = "live admission"
	return c
}

func recordDreamCandidate(c dreamCandidate) error {
	path := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	err = enc.Encode(c)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func rejectOnAdmissionLogError(c dreamCandidate, err error) dreamCandidate {
	if err == nil {
		return c
	}
	c.Accepted = false
	c.Reason = "admission log failed: " + err.Error()
	return c
}

func attachDreamCounterfactual(iw *InnerWorld, c dreamCandidate) dreamCandidate {
	if iw == nil || strings.TrimSpace(c.Text) == "" {
		return c
	}
	before := iw.GetSnapshot()
	scratch := NewInnerWorld()
	applySnapshot(scratch, before)
	scratch.Start(false)
	syncScratchProcesses(scratch, before)
	analysis := scratch.ProcessText(c.Text)
	syncEmotionalPosition(scratch)
	after := scratch.GetSnapshot()
	scratch.Stop()

	c.Counterfactual = &dreamCounterfactual{
		Schema:        "arianna.dream_counterfactual.v1",
		Target:        "inner_world",
		PreStateHash:  hashSnapshot(before),
		PostStateHash: hashSnapshot(after),
		Delta:         diffSnapshot(before, after),
		Analysis:      analysis,
		Text:          measureDreamText(c.Text),
	}
	return c
}

func applySnapshot(iw *InnerWorld, s Snapshot) {
	iw.State.mu.Lock()
	defer iw.State.mu.Unlock()
	iw.State.Arousal = s.Arousal
	iw.State.Valence = s.Valence
	iw.State.Entropy = s.Entropy
	iw.State.Coherence = s.Coherence
	iw.State.TraumaLevel = s.TraumaLevel
	iw.State.LoopCount = s.LoopCount
	iw.State.AbstractionDepth = s.AbstractionDepth
	iw.State.SelfRefCount = s.SelfRefCount
	iw.State.DriftDirection = s.DriftDirection
	iw.State.DriftSpeed = s.DriftSpeed
	iw.State.DriftTarget = s.DriftTarget
	iw.State.MemoryPressure = s.MemoryPressure
	iw.State.FocusStrength = s.FocusStrength
	iw.State.WanderPull = s.WanderPull
	iw.State.ProphecyDebt = s.ProphecyDebt
	iw.State.DestinyPull = s.DestinyPull
	iw.State.WormholeChance = s.WormholeChance
}

func syncScratchProcesses(iw *InnerWorld, s Snapshot) {
	if ed := iw.GetEmotionalDrift(); ed != nil {
		ed.Resync()
	}
	if aw := iw.GetAttentionWandering(); aw != nil {
		aw.focusStrength = s.FocusStrength
	}
}

func syncEmotionalPosition(iw *InnerWorld) {
	ed := iw.GetEmotionalDrift()
	if ed == nil {
		return
	}
	pos := ed.GetPosition()
	iw.State.mu.Lock()
	iw.State.Valence = pos.Valence
	iw.State.Arousal = pos.Arousal
	iw.State.mu.Unlock()
}

func hashSnapshot(s Snapshot) string {
	h := sha256.New()
	fmt.Fprintf(h, "a=%.9g|v=%.9g|e=%.9g|c=%.9g|t=%.9g|l=%d|ad=%d|sr=%d|dd=%.9g|ds=%.9g|dt=%s|m=%.9g|f=%.9g|w=%.9g|p=%.9g|dp=%.9g|wc=%.9g",
		s.Arousal, s.Valence, s.Entropy, s.Coherence, s.TraumaLevel,
		s.LoopCount, s.AbstractionDepth, s.SelfRefCount,
		s.DriftDirection, s.DriftSpeed, s.DriftTarget,
		s.MemoryPressure, s.FocusStrength, s.WanderPull,
		s.ProphecyDebt, s.DestinyPull, s.WormholeChance)
	return hex.EncodeToString(h.Sum(nil)[:8])
}

func diffSnapshot(before, after Snapshot) dreamSnapshotDelta {
	return dreamSnapshotDelta{
		Arousal:          after.Arousal - before.Arousal,
		Valence:          after.Valence - before.Valence,
		Entropy:          after.Entropy - before.Entropy,
		Coherence:        after.Coherence - before.Coherence,
		TraumaLevel:      after.TraumaLevel - before.TraumaLevel,
		LoopCount:        after.LoopCount - before.LoopCount,
		AbstractionDepth: after.AbstractionDepth - before.AbstractionDepth,
		SelfRefCount:     after.SelfRefCount - before.SelfRefCount,
		MemoryPressure:   after.MemoryPressure - before.MemoryPressure,
		FocusStrength:    after.FocusStrength - before.FocusStrength,
		WanderPull:       after.WanderPull - before.WanderPull,
		ProphecyDebt:     after.ProphecyDebt - before.ProphecyDebt,
	}
}

func measureDreamText(text string) dreamTextMetrics {
	m := dreamTextMetrics{Bytes: len(text), Words: len(strings.Fields(text))}
	terms := make([]string, 0, 4)
	lower := strings.ToLower(text)
	for _, term := range []string{"oleg", "олег", "user", "assistant"} {
		if strings.Contains(lower, term) {
			terms = append(terms, term)
		}
	}
	for _, r := range text {
		m.Runes++
		switch {
		case unicode.In(r, unicode.Cyrillic):
			m.CyrillicRunes++
		case unicode.Is(unicode.Latin, r):
			m.LatinRunes++
		}
	}
	switch {
	case m.CyrillicRunes > 0 && m.LatinRunes > 0:
		m.LanguageHint = "mixed"
	case m.CyrillicRunes > 0:
		m.LanguageHint = "ru"
	case m.LatinRunes > 0:
		m.LanguageHint = "en"
	default:
		m.LanguageHint = "unknown"
	}
	m.RecipientTerms = terms
	return m
}

func admitDreamToInnerWorld(iw *InnerWorld, r *dreamResult, trigger string) bool {
	if r == nil || strings.TrimSpace(r.dream) == "" {
		return false
	}
	c := r.candidate
	if c.Schema == "" {
		c = newDreamCandidate("nano", trigger, "", r.frag, r.dream, nil)
	}
	c.Trigger = trigger
	c = decideDreamCandidate(c)
	c = attachDreamCounterfactual(iw, c)
	c = rejectOnAdmissionLogError(c, recordDreamCandidate(c))
	r.candidate = c
	if !c.Accepted {
		return false
	}
	iw.ProcessText(r.dream)
	return true
}

func (r dreamResult) admitted() bool {
	return r.candidate.Accepted
}

func (r dreamResult) admissionLabel() string {
	if r.candidate.Schema == "" {
		return "untyped"
	}
	if r.candidate.Accepted {
		return "accepted"
	}
	if r.candidate.Reason != "" {
		return r.candidate.Reason
	}
	return "rejected"
}
