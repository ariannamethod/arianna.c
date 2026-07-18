package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"
	"time"
)

const (
	dreamAdmissionLive   = "live"
	dreamAdmissionShadow = "shadow"
)

// dreamCandidate is the typed boundary between observing a subconscious trace
// and letting it mutate Arianna.c state. Default live mode preserves existing
// behavior; AM_DREAM_ADMISSION=shadow records the candidate and rejects mutation.
type dreamCandidate struct {
	Schema    string
	RunID     string
	Mode      string
	Source    string
	Trigger   string
	Seed      string
	Fragment  string
	Text      string
	Kind      string
	Cells     int
	Questions int
	Accepted  bool
	Reason    string
	Created   time.Time
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
