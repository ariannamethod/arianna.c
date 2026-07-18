package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
