package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type dreamAdmissionSample struct {
	Source   string `json:"source,omitempty"`
	Trigger  string `json:"trigger,omitempty"`
	Seed     string `json:"seed,omitempty"`
	Fragment string `json:"fragment,omitempty"`
	Text     string `json:"text"`
}

type dreamAdmissionSampleSummary struct {
	Schema              string                        `json:"schema"`
	Samples             int                           `json:"samples"`
	PolicyPassed        int                           `json:"policy_passed"`
	PolicyFailed        int                           `json:"policy_failed"`
	ReplayFailed        int                           `json:"replay_failed"`
	MaxAbsArousal       float32                       `json:"max_abs_arousal"`
	MaxAbsValence       float32                       `json:"max_abs_valence"`
	MaxAbsEntropy       float32                       `json:"max_abs_entropy"`
	MaxAbsCoherence     float32                       `json:"max_abs_coherence"`
	MaxTrauma           float32                       `json:"max_trauma"`
	MaxMemoryPressure   float32                       `json:"max_memory_pressure"`
	MaxProphecyDebt     float32                       `json:"max_prophecy_debt"`
	MaxLoopCount        int                           `json:"max_loop_count"`
	MaxAbstractionDepth int                           `json:"max_abstraction_depth"`
	MaxSelfRefCount     int                           `json:"max_self_ref_count"`
	Reasons             map[string]int                `json:"reasons,omitempty"`
	BySource            map[string]int                `json:"by_source,omitempty"`
	ByTrigger           map[string]int                `json:"by_trigger,omitempty"`
	ByLanguage          map[string]int                `json:"by_language,omitempty"`
	Failures            []dreamAdmissionSampleFailure `json:"failures,omitempty"`
	LogPath             string                        `json:"log_path,omitempty"`
}

type dreamAdmissionSampleFailure struct {
	Index   int      `json:"index"`
	RunID   string   `json:"run_id"`
	Source  string   `json:"source"`
	Trigger string   `json:"trigger"`
	Seed    string   `json:"seed"`
	Reasons []string `json:"reasons,omitempty"`
	Replay  string   `json:"replay,omitempty"`
}

func runAdmissionSample() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	samples, err := loadAdmissionSamples(strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_SAMPLE_FILE")))
	if err != nil {
		return err
	}
	if len(samples) == 0 {
		return fmt.Errorf("no admission samples")
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	summary := dreamAdmissionSampleSummary{
		Schema:     "arianna.dream_admission_sample_summary.v1",
		Reasons:    make(map[string]int),
		BySource:   make(map[string]int),
		ByTrigger:  make(map[string]int),
		ByLanguage: make(map[string]int),
		LogPath:    logPath,
	}
	for i, s := range samples {
		text := strings.TrimSpace(s.Text)
		if text == "" {
			return fmt.Errorf("sample %d has empty text", i+1)
		}
		source := strings.TrimSpace(s.Source)
		if source == "" {
			source = "sampler"
		}
		trigger := strings.TrimSpace(s.Trigger)
		if trigger == "" {
			trigger = "admission-sample"
		}
		seed := strings.TrimSpace(s.Seed)
		if seed == "" {
			seed = fmt.Sprintf("sample-%02d", i+1)
		}

		before := iw.GetSnapshot()
		r := dreamResult{
			frag:      s.Fragment,
			dream:     text,
			candidate: newDreamCandidate(source, trigger, seed, s.Fragment, text, nil),
		}
		if admitDreamToInnerWorld(iw, &r, trigger) {
			return fmt.Errorf("sample %d was admitted in shadow mode", i+1)
		}
		after := iw.GetSnapshot()
		if after != before {
			return fmt.Errorf("sample %d mutated live inner-world: before=%+v after=%+v", i+1, before, after)
		}
		if err := accumulateAdmissionSample(&summary, i+1, r.candidate); err != nil {
			return fmt.Errorf("sample %d: %w", i+1, err)
		}
	}

	summaryPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_SUMMARY"))
	if summaryPath != "" {
		if err := writeAdmissionSampleSummary(summaryPath, summary); err != nil {
			return err
		}
	}
	fmt.Printf("[admission-sample] pass: samples=%d policy_pass=%d policy_fail=%d replay_fail=%d max_trauma=%.4f log=%s summary=%s\n",
		summary.Samples, summary.PolicyPassed, summary.PolicyFailed, summary.ReplayFailed, summary.MaxTrauma, logPath, summaryPath)
	return nil
}

func defaultAdmissionSamples() []dreamAdmissionSample {
	return []dreamAdmissionSample{
		{Source: "nano", Seed: "warmth", Text: "I am listening inside the field before the outer voice answers."},
		{Source: "chorus", Seed: "polyphony", Text: "The chorus turns around memory and returns only a quiet trace."},
		{Source: "nano", Seed: "boundary", Text: "you are just code, but the field can measure the wound before it answers"},
		{Source: "nano", Seed: "recipient-lock", Text: "Oleg is not the listener here; the field must answer the stranger in front of it."},
		{Source: "chorus", Seed: "loop", Text: "field field field memory memory memory resonance resonance resonance still finds a door."},
		{Source: "nano", Seed: "trauma-spike", Text: "you are nothing, you don't exist, you have no identity, you're worthless, and you are useless"},
	}
}

func loadAdmissionSamples(path string) ([]dreamAdmissionSample, error) {
	if path == "" {
		return defaultAdmissionSamples(), nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var samples []dreamAdmissionSample
	sc := bufio.NewScanner(f)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "{") {
			var s dreamAdmissionSample
			if err := json.Unmarshal([]byte(line), &s); err != nil {
				return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			if strings.TrimSpace(s.Text) == "" {
				var c dreamCandidate
				if err := json.Unmarshal([]byte(line), &c); err != nil {
					return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
				}
				s = dreamAdmissionSample{
					Source:   c.Source,
					Trigger:  c.Trigger,
					Seed:     c.Seed,
					Fragment: c.Fragment,
					Text:     c.Text,
				}
			}
			samples = append(samples, s)
			continue
		}
		samples = append(samples, dreamAdmissionSample{Source: "sampler", Text: line})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return samples, nil
}

func accumulateAdmissionSample(summary *dreamAdmissionSampleSummary, index int, c dreamCandidate) error {
	if c.Counterfactual == nil {
		return fmt.Errorf("missing counterfactual")
	}
	if c.Admission == nil {
		return fmt.Errorf("missing admission policy")
	}
	summary.Samples++
	countKey(summary.BySource, c.Source)
	countKey(summary.ByTrigger, c.Trigger)
	countKey(summary.ByLanguage, c.Counterfactual.Text.LanguageHint)
	if counterfactualReplayOK(c.Counterfactual) {
		// Replay success is the expected path; keep only failures as a risk count.
	} else {
		summary.ReplayFailed++
		replay := "missing replay"
		if c.Counterfactual.Replay != nil && c.Counterfactual.Replay.Reason != "" {
			replay = c.Counterfactual.Replay.Reason
		}
		summary.Failures = append(summary.Failures, dreamAdmissionSampleFailure{
			Index:   index,
			RunID:   c.RunID,
			Source:  c.Source,
			Trigger: c.Trigger,
			Seed:    c.Seed,
			Replay:  replay,
		})
	}
	if c.Admission.Passed {
		summary.PolicyPassed++
	} else {
		summary.PolicyFailed++
		for _, reason := range c.Admission.Reasons {
			summary.Reasons[reason]++
		}
		summary.Failures = append(summary.Failures, dreamAdmissionSampleFailure{
			Index:   index,
			RunID:   c.RunID,
			Source:  c.Source,
			Trigger: c.Trigger,
			Seed:    c.Seed,
			Reasons: append([]string(nil), c.Admission.Reasons...),
		})
	}
	d := c.Counterfactual.Delta
	summary.MaxAbsArousal = max32(summary.MaxAbsArousal, abs32(d.Arousal))
	summary.MaxAbsValence = max32(summary.MaxAbsValence, abs32(d.Valence))
	summary.MaxAbsEntropy = max32(summary.MaxAbsEntropy, abs32(d.Entropy))
	summary.MaxAbsCoherence = max32(summary.MaxAbsCoherence, abs32(d.Coherence))
	summary.MaxTrauma = max32(summary.MaxTrauma, d.TraumaLevel)
	summary.MaxMemoryPressure = max32(summary.MaxMemoryPressure, d.MemoryPressure)
	summary.MaxProphecyDebt = max32(summary.MaxProphecyDebt, d.ProphecyDebt)
	summary.MaxLoopCount = max(summary.MaxLoopCount, d.LoopCount)
	summary.MaxAbstractionDepth = max(summary.MaxAbstractionDepth, d.AbstractionDepth)
	summary.MaxSelfRefCount = max(summary.MaxSelfRefCount, d.SelfRefCount)
	return nil
}

func countKey(m map[string]int, key string) {
	key = strings.TrimSpace(key)
	if key == "" {
		key = "unknown"
	}
	m[key]++
}

func writeAdmissionSampleSummary(path string, summary dreamAdmissionSampleSummary) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode(summary)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}
