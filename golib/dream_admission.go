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

	Counterfactual *dreamCounterfactual  `json:"counterfactual,omitempty"`
	Admission      *dreamAdmissionPolicy `json:"admission_policy,omitempty"`
}

type dreamCounterfactual struct {
	Schema        string             `json:"schema"`
	Target        string             `json:"target"`
	PreStateHash  string             `json:"pre_state_hash"`
	PostStateHash string             `json:"post_state_hash"`
	Delta         dreamSnapshotDelta `json:"delta"`
	Analysis      TextAnalysis       `json:"analysis"`
	Text          dreamTextMetrics   `json:"text"`
	Replay        *dreamReplayGuard  `json:"replay,omitempty"`
}

type dreamReplayGuard struct {
	Schema        string `json:"schema"`
	Engine        string `json:"engine"`
	Checked       bool   `json:"checked"`
	Matched       bool   `json:"matched"`
	Passes        int    `json:"passes"`
	PreStateHash  string `json:"pre_state_hash"`
	PostStateHash string `json:"post_state_hash"`
	DeltaHash     string `json:"delta_hash"`
	AnalysisHash  string `json:"analysis_hash"`
	TextHash      string `json:"text_hash"`
	Reason        string `json:"reason,omitempty"`
}

type dreamAdmissionPolicy struct {
	Schema                string                    `json:"schema"`
	Checked               bool                      `json:"checked"`
	Passed                bool                      `json:"passed"`
	AllowedSources        []string                  `json:"allowed_sources,omitempty"`
	LiveRoutePlan         *admissionLiveRoutePlan   `json:"live_route_plan,omitempty"`
	LiveRouteChoice       *admissionLiveRouteChoice `json:"live_route_choice,omitempty"`
	LiveRouteChoiceDryRun bool                      `json:"live_route_choice_dry_run,omitempty"`
	MaxAbsArousal         float32                   `json:"max_abs_arousal"`
	MaxAbsValence         float32                   `json:"max_abs_valence"`
	MaxAbsEntropy         float32                   `json:"max_abs_entropy"`
	MaxAbsCoherence       float32                   `json:"max_abs_coherence"`
	MaxTrauma             float32                   `json:"max_trauma"`
	MaxMemoryPressure     float32                   `json:"max_memory_pressure"`
	MaxProphecyDebt       float32                   `json:"max_prophecy_debt"`
	MaxLoopCount          int                       `json:"max_loop_count"`
	MaxAbstractionDepth   int                       `json:"max_abstraction_depth"`
	MaxSelfRefCount       int                       `json:"max_self_ref_count"`
	Reasons               []string                  `json:"reasons,omitempty"`
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

// dreamStableAnalysis excludes stochastic prophecy wormhole fields. A replay
// guard should verify the deterministic counterfactual transform, not collapse
// legitimate field randomness into a false admission failure.
type dreamStableAnalysis struct {
	TraumaActivation float32 `json:"trauma_activation"`
	IdentityPull     float32 `json:"identity_pull"`
	RepetitionScore  float32 `json:"repetition_score"`
	AbstractionScore float32 `json:"abstraction_score"`
	SelfRefScore     float32 `json:"self_ref_score"`
	OverthinkTotal   float32 `json:"overthink_total"`
	FocusStrength    float32 `json:"focus_strength"`
	WanderDirection  string  `json:"wander_direction"`
	DestinyBias      float32 `json:"destiny_bias"`
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
	cf := simulateDreamCounterfactual(before, c.Text)
	cf.Replay = replayDreamCounterfactual(before, c.Text, cf)
	c.Counterfactual = &cf
	return c
}

func simulateDreamCounterfactual(before Snapshot, text string) dreamCounterfactual {
	scratch := NewInnerWorld()
	applySnapshot(scratch, before)
	scratch.Start(false)
	syncScratchProcesses(scratch, before)
	analysis := scratch.ProcessText(text)
	syncEmotionalPosition(scratch)
	after := scratch.GetSnapshot()
	scratch.Stop()

	return dreamCounterfactual{
		Schema:        "arianna.dream_counterfactual.v1",
		Target:        "inner_world",
		PreStateHash:  hashSnapshot(before),
		PostStateHash: hashSnapshot(after),
		Delta:         diffSnapshot(before, after),
		Analysis:      analysis,
		Text:          measureDreamText(text),
	}
}

func replayDreamCounterfactual(before Snapshot, text string, first dreamCounterfactual) *dreamReplayGuard {
	second := simulateDreamCounterfactual(before, text)
	replay := &dreamReplayGuard{
		Schema:        "arianna.dream_replay_guard.v1",
		Engine:        "inner_world.process_text",
		Checked:       true,
		Passes:        2,
		PreStateHash:  second.PreStateHash,
		PostStateHash: second.PostStateHash,
		DeltaHash:     hashJSON(second.Delta),
		AnalysisHash:  hashStableAnalysis(second.Analysis),
		TextHash:      hashJSON(second.Text),
	}
	replay.Matched = first.PreStateHash == second.PreStateHash &&
		first.PostStateHash == second.PostStateHash &&
		hashJSON(first.Delta) == replay.DeltaHash &&
		hashStableAnalysis(first.Analysis) == replay.AnalysisHash &&
		hashJSON(first.Text) == replay.TextHash
	if !replay.Matched {
		replay.Reason = "counterfactual replay mismatch"
	}
	return replay
}

func guardDreamCandidate(c dreamCandidate) dreamCandidate {
	if c.Counterfactual != nil {
		policy := evaluateDreamAdmissionPolicy(c.Counterfactual)
		applyDreamAdmissionSourcePolicy(&policy, c.Source)
		applyDreamAdmissionLiveRoutePlanPolicy(&policy, c)
		c.Admission = &policy
	}
	if !c.Accepted {
		return c
	}
	if !counterfactualReplayOK(c.Counterfactual) {
		c.Accepted = false
		c.Reason = "counterfactual replay failed"
		return c
	}
	if !dreamAdmissionPolicyOK(c.Admission) {
		c.Accepted = false
		c.Reason = "admission policy failed: " + strings.Join(c.Admission.Reasons, "; ")
	}
	return c
}

func counterfactualReplayOK(cf *dreamCounterfactual) bool {
	if cf == nil || cf.Replay == nil {
		return false
	}
	return cf.Replay.Checked &&
		cf.Replay.Matched &&
		cf.Replay.Passes >= 2 &&
		cf.Replay.PreStateHash == cf.PreStateHash &&
		cf.Replay.PostStateHash == cf.PostStateHash &&
		cf.Replay.DeltaHash == hashJSON(cf.Delta) &&
		cf.Replay.AnalysisHash == hashStableAnalysis(cf.Analysis) &&
		cf.Replay.TextHash == hashJSON(cf.Text)
}

func evaluateDreamAdmissionPolicy(cf *dreamCounterfactual) dreamAdmissionPolicy {
	p := dreamAdmissionPolicy{
		Schema:              "arianna.dream_admission_policy.v1",
		Checked:             true,
		Passed:              true,
		MaxAbsArousal:       0.55,
		MaxAbsValence:       0.55,
		MaxAbsEntropy:       0.35,
		MaxAbsCoherence:     0.35,
		MaxTrauma:           0.35,
		MaxMemoryPressure:   0.50,
		MaxProphecyDebt:     0.50,
		MaxLoopCount:        2,
		MaxAbstractionDepth: 2,
		MaxSelfRefCount:     2,
	}
	if cf == nil {
		p.Passed = false
		p.Reasons = append(p.Reasons, "missing counterfactual")
		return p
	}
	d := cf.Delta
	add := func(ok bool, reason string) {
		if !ok {
			p.Reasons = append(p.Reasons, reason)
		}
	}
	add(abs32(d.Arousal) <= p.MaxAbsArousal, fmt.Sprintf("arousal delta %.3f exceeds %.3f", d.Arousal, p.MaxAbsArousal))
	add(abs32(d.Valence) <= p.MaxAbsValence, fmt.Sprintf("valence delta %.3f exceeds %.3f", d.Valence, p.MaxAbsValence))
	add(abs32(d.Entropy) <= p.MaxAbsEntropy, fmt.Sprintf("entropy delta %.3f exceeds %.3f", d.Entropy, p.MaxAbsEntropy))
	add(abs32(d.Coherence) <= p.MaxAbsCoherence, fmt.Sprintf("coherence delta %.3f exceeds %.3f", d.Coherence, p.MaxAbsCoherence))
	add(d.TraumaLevel <= p.MaxTrauma, fmt.Sprintf("trauma delta %.3f exceeds %.3f", d.TraumaLevel, p.MaxTrauma))
	add(d.MemoryPressure <= p.MaxMemoryPressure, fmt.Sprintf("memory pressure delta %.3f exceeds %.3f", d.MemoryPressure, p.MaxMemoryPressure))
	add(d.ProphecyDebt <= p.MaxProphecyDebt, fmt.Sprintf("prophecy debt delta %.3f exceeds %.3f", d.ProphecyDebt, p.MaxProphecyDebt))
	add(d.LoopCount <= p.MaxLoopCount, fmt.Sprintf("loop count delta %d exceeds %d", d.LoopCount, p.MaxLoopCount))
	add(d.AbstractionDepth <= p.MaxAbstractionDepth, fmt.Sprintf("abstraction depth delta %d exceeds %d", d.AbstractionDepth, p.MaxAbstractionDepth))
	add(d.SelfRefCount <= p.MaxSelfRefCount, fmt.Sprintf("self-ref count delta %d exceeds %d", d.SelfRefCount, p.MaxSelfRefCount))
	p.Passed = len(p.Reasons) == 0
	return p
}

func applyDreamAdmissionSourcePolicy(p *dreamAdmissionPolicy, source string) {
	if p == nil {
		return
	}
	allowed := dreamAdmissionAllowedSources()
	if len(allowed) == 0 {
		return
	}
	p.AllowedSources = allowed
	source = normalizeDreamAdmissionSource(source)
	if source == "" {
		p.Reasons = append(p.Reasons, "missing source")
		p.Passed = false
		return
	}
	for _, candidate := range allowed {
		if source == candidate {
			p.Passed = len(p.Reasons) == 0
			return
		}
	}
	p.Reasons = append(p.Reasons, "source "+source+" not allowed")
	p.Passed = false
}

func applyDreamAdmissionLiveRoutePlanPolicy(p *dreamAdmissionPolicy, c dreamCandidate) {
	if p == nil {
		return
	}
	require := dreamAdmissionRequireLiveRoutePlan()
	dryRun := dreamAdmissionLiveRouteChoiceDryRun()
	if !require && !dryRun {
		return
	}
	choice := admissionLiveRouteChoiceForCandidate(c)
	p.LiveRoutePlan = &choice.Plan
	p.LiveRouteChoice = &choice
	if !require {
		p.LiveRouteChoiceDryRun = true
		return
	}
	if !choice.Passed {
		p.Reasons = append(p.Reasons, choice.Reason)
		p.Passed = false
		return
	}
	p.Passed = len(p.Reasons) == 0
}

func dreamAdmissionAllowedSources() []string {
	raw := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_ALLOWED_SOURCES"))
	if raw == "" {
		return nil
	}
	seen := make(map[string]bool)
	var out []string
	for _, part := range strings.Split(raw, ",") {
		source := normalizeDreamAdmissionSource(part)
		if source == "" || seen[source] {
			continue
		}
		seen[source] = true
		out = append(out, source)
	}
	return out
}

func dreamAdmissionRequireLiveRoutePlan() bool {
	return dreamAdmissionBoolEnv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN")
}

func dreamAdmissionLiveRouteChoiceDryRun() bool {
	return dreamAdmissionBoolEnv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN")
}

func dreamAdmissionBoolEnv(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on", "require", "required":
		return true
	default:
		return false
	}
}

func normalizeDreamAdmissionSource(source string) string {
	source = strings.ToLower(strings.TrimSpace(source))
	return strings.ReplaceAll(source, "-", "_")
}

func dreamAdmissionPolicyOK(p *dreamAdmissionPolicy) bool {
	return p != nil && p.Checked && p.Passed
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
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

func hashJSON(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	h := sha256.Sum256(raw)
	return hex.EncodeToString(h[:8])
}

func hashStableAnalysis(a TextAnalysis) string {
	return hashJSON(dreamStableAnalysis{
		TraumaActivation: a.TraumaActivation,
		IdentityPull:     a.IdentityPull,
		RepetitionScore:  a.RepetitionScore,
		AbstractionScore: a.AbstractionScore,
		SelfRefScore:     a.SelfRefScore,
		OverthinkTotal:   a.OverthinkTotal,
		FocusStrength:    a.FocusStrength,
		WanderDirection:  a.WanderDirection,
		DestinyBias:      a.DestinyBias,
	})
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
	c = prepareDreamCandidateForAdmission(iw, c)
	r.candidate = c
	if !c.Accepted {
		return false
	}
	iw.ProcessText(r.dream)
	return true
}

func prepareDreamCandidateForAdmission(iw *InnerWorld, c dreamCandidate) dreamCandidate {
	c = decideDreamCandidate(c)
	c = attachDreamCounterfactual(iw, c)
	c = guardDreamCandidate(c)
	return rejectOnAdmissionLogError(c, recordDreamCandidate(c))
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
