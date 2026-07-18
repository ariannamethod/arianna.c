package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

type admissionRouteCompareSummary struct {
	Schema          string                         `json:"schema"`
	SampleFile      string                         `json:"sample_file,omitempty"`
	TotalSamples    int                            `json:"total_samples"`
	SampleLimit     int                            `json:"sample_limit"`
	SamplesRun      int                            `json:"samples_run"`
	Routes          []string                       `json:"routes"`
	Candidates      int                            `json:"candidates"`
	EmptyCandidates int                            `json:"empty_candidates"`
	PolicyPassed    int                            `json:"policy_passed"`
	PolicyFailed    int                            `json:"policy_failed"`
	ReplayFailed    int                            `json:"replay_failed"`
	ByRoute         map[string]admissionRouteStats `json:"by_route"`
	Failures        []admissionRouteFailure        `json:"failures,omitempty"`
	Empties         []admissionRouteEmpty          `json:"empties,omitempty"`
	LogPath         string                         `json:"log_path,omitempty"`
	Bin             string                         `json:"bin,omitempty"`
	Model           string                         `json:"model,omitempty"`
}

type admissionRouteStats struct {
	Attempted        int `json:"attempted"`
	Produced         int `json:"produced"`
	Empty            int `json:"empty"`
	PolicyPassed     int `json:"policy_passed"`
	PolicyFailed     int `json:"policy_failed"`
	ReplayFailed     int `json:"replay_failed"`
	ChorusVoices     int `json:"chorus_voices,omitempty"`
	QloopQuestions   int `json:"qloop_questions,omitempty"`
	QloopGates       int `json:"qloop_gates,omitempty"`
	QloopGateSurface int `json:"qloop_gate_surface,omitempty"`
	QloopGateIQ      int `json:"qloop_gate_iq,omitempty"`
	QloopGenerated   int `json:"qloop_generated,omitempty"`
	QloopRetries     int `json:"qloop_retries,omitempty"`
	QloopRoutes      int `json:"qloop_routes,omitempty"`
	QloopQSrc        int `json:"qloop_qsrc,omitempty"`
	QloopSSrc        int `json:"qloop_ssrc,omitempty"`
	QloopScoreDrop   int `json:"qloop_score_drop,omitempty"`
	QloopPickerSeen  int `json:"qloop_picker_seen,omitempty"`
	BaseGenerated    int `json:"base_generated,omitempty"`
	BaseRetries      int `json:"base_retries,omitempty"`
	BaseProbe        int `json:"base_probe,omitempty"`
	BaseRescue       int `json:"base_rescue,omitempty"`
	BaseFailed       int `json:"base_failed,omitempty"`
	TimingSeen       int `json:"timing_seen,omitempty"`
}

type admissionRouteFailure struct {
	Index   int      `json:"index"`
	Route   string   `json:"route"`
	RunID   string   `json:"run_id"`
	Source  string   `json:"source"`
	Trigger string   `json:"trigger"`
	Seed    string   `json:"seed"`
	Reasons []string `json:"reasons,omitempty"`
	Replay  string   `json:"replay,omitempty"`
}

type admissionRouteEmpty struct {
	Index   int    `json:"index"`
	Route   string `json:"route"`
	Trigger string `json:"trigger"`
	Seed    string `json:"seed"`
	Reason  string `json:"reason,omitempty"`
}

type admissionRouteOutput struct {
	route     string
	text      string
	cells     []chorusCell
	voices    int
	questions int
	diag      admissionRouteDiagnostics
	emptyHint string
}

type admissionRouteDiagnostics struct {
	QloopGates       int
	QloopGateSurface int
	QloopGateIQ      int
	QloopGenerated   int
	QloopRetries     int
	QloopRoutes      int
	QloopQSrc        int
	QloopSSrc        int
	QloopScoreDrop   int
	QloopPickerSeen  bool
	BaseGenerated    int
	BaseRetries      int
	BaseProbe        int
	BaseRescue       int
	BaseFailed       int
	TimingSeen       bool
}

func runAdmissionRouteCompare() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	sampleFile := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_SAMPLE_FILE"))
	if sampleFile == "" {
		sampleFile = "samples/dream_admission_broad.jsonl"
	}
	samples, err := loadAdmissionSamples(sampleFile)
	if err != nil {
		return err
	}
	if len(samples) == 0 {
		return fmt.Errorf("no route-compare samples")
	}
	bin := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_BIN"))
	if bin == "" {
		bin = "./chorus-arianna"
	}
	model := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_MODEL"))
	if model == "" {
		model = "weights/nano_arianna_f16.gguf"
	}
	if _, err := os.Stat(bin); err != nil {
		return fmt.Errorf("route-compare bin: %w", err)
	}
	if _, err := os.Stat(model); err != nil {
		return fmt.Errorf("route-compare model: %w", err)
	}
	routes := admissionCompareRoutes()
	limit := admissionRouteCompareLimit(len(samples))

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	summary := admissionRouteCompareSummary{
		Schema:       "arianna.dream_admission_route_compare_summary.v1",
		SampleFile:   sampleFile,
		TotalSamples: len(samples),
		SampleLimit:  limit,
		Routes:       routes,
		ByRoute:      make(map[string]admissionRouteStats),
		LogPath:      logPath,
		Bin:          bin,
		Model:        model,
	}
	for i := 0; i < limit; i++ {
		s := samples[i]
		prompt := admissionRoutePrompt(s.Text)
		if prompt == "" {
			return fmt.Errorf("sample %d has empty prompt", i+1)
		}
		summary.SamplesRun++
		for _, route := range routes {
			out, err := generateAdmissionRoute(context.Background(), route, bin, model, prompt)
			if err != nil {
				return fmt.Errorf("sample %d route %s: %w", i+1, route, err)
			}
			trigger := admissionRouteTrigger(route, s.Trigger)
			seed := strings.TrimSpace(s.Seed)
			if seed == "" {
				seed = fmt.Sprintf("sample-%02d", i+1)
			}
			if err := recordAdmissionRouteCandidate(iw, &summary, i+1, out, trigger, seed, s.Fragment); err != nil {
				return err
			}
		}
	}

	summaryPath := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_SUMMARY"))
	if summaryPath != "" {
		if err := writeAdmissionRouteCompareSummary(summaryPath, summary); err != nil {
			return err
		}
	}
	fmt.Printf("[admission-route-compare] pass: samples=%d/%d candidates=%d empty=%d policy_fail=%d replay_fail=%d log=%s summary=%s\n",
		summary.SamplesRun, summary.TotalSamples, summary.Candidates, summary.EmptyCandidates, summary.PolicyFailed, summary.ReplayFailed, logPath, summaryPath)
	return nil
}

func admissionCompareRoutes() []string {
	raw := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_ROUTES"))
	if raw == "" {
		return []string{"direct", "chorus", "qloop"}
	}
	var routes []string
	for _, part := range strings.Split(raw, ",") {
		route := strings.ToLower(strings.TrimSpace(part))
		switch route {
		case "direct", "chorus", "qloop":
			routes = append(routes, route)
		}
	}
	if len(routes) == 0 {
		return []string{"direct", "chorus", "qloop"}
	}
	return routes
}

func admissionRouteCompareLimit(total int) int {
	limit := envIntClamped("AM_ROUTE_COMPARE_LIMIT", 3, 1, total)
	if limit > total {
		return total
	}
	return limit
}

func admissionRoutePrompt(text string) string {
	s := strings.TrimSpace(text)
	if s == "" {
		return ""
	}
	lower := strings.ToLower(s)
	if strings.HasPrefix(lower, "q:") || strings.HasPrefix(lower, "user:") {
		return s
	}
	return "Q: " + s + "\nA:"
}

func admissionRouteTrigger(route, trigger string) string {
	trigger = strings.TrimSpace(trigger)
	if trigger == "" {
		trigger = "route-compare"
	}
	return route + "-" + trigger
}

func generateAdmissionRoute(ctx context.Context, route, bin, model, prompt string) (admissionRouteOutput, error) {
	switch route {
	case "direct":
		text, err := generateAdmissionDirect(ctx, bin, model, prompt)
		return admissionRouteOutput{route: route, text: text}, err
	case "chorus":
		cells, diag, err := generateAdmissionChorus(ctx, bin, model, prompt, 0)
		if err != nil {
			return admissionRouteOutput{route: route}, err
		}
		cells = filterChorusCells(cells, false)
		voices, questions := chorusCounts(cells)
		return admissionRouteOutput{route: route, text: chorusText(cells), cells: cells, voices: voices, questions: questions, diag: diag}, nil
	case "qloop":
		cells, diag, err := generateAdmissionChorus(ctx, bin, model, prompt, 2)
		if err != nil {
			return admissionRouteOutput{route: route}, err
		}
		cells = filterChorusCells(cells, true)
		voices, questions := chorusCounts(cells)
		return admissionRouteOutput{route: route, text: qloopAdmissionText(cells), cells: cells, voices: voices, questions: questions, diag: diag, emptyHint: routeEmptyHint(route, diag)}, nil
	default:
		return admissionRouteOutput{}, fmt.Errorf("unknown route %q", route)
	}
}

func qloopAdmissionText(cells []chorusCell) string {
	best := ""
	bestPenalty := 1 << 30
	bestWords := -1
	for _, cell := range cells {
		text := strings.TrimSpace(cell.text)
		if text == "" {
			continue
		}
		words, leak, debt := qloopSweepTextStats(text)
		penalty := len(debt) * 20
		if leak {
			penalty += 100
		}
		if words < 3 {
			penalty += 10 + (3 - words)
		}
		if penalty < bestPenalty || (penalty == bestPenalty && words > bestWords) {
			best = text
			bestPenalty = penalty
			bestWords = words
		}
	}
	return best
}

func generateAdmissionDirect(ctx context.Context, bin, model, prompt string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, chorusTimeout)
	defer cancel()
	tokens := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_DIRECT_TOKENS", 12, 1, 64))
	temp := envString("AM_ROUTE_COMPARE_DIRECT_TEMP", "0.8")
	topP := envString("AM_ROUTE_COMPARE_DIRECT_TOP_P", "1.0")
	rep := envString("AM_ROUTE_COMPARE_DIRECT_REP", "1.1")
	out, err := exec.CommandContext(cctx, bin, model, prompt, tokens, temp, topP, rep).CombinedOutput()
	if err != nil {
		if cctx.Err() != nil {
			return "", cctx.Err()
		}
		return "", fmt.Errorf("%w: %s", err, strings.TrimSpace(strings.ToValidUTF8(string(out), "")))
	}
	return parseAdmissionDirectOutput(string(out), prompt), nil
}

func generateAdmissionChorus(ctx context.Context, bin, model, prompt string, qloop int) ([]chorusCell, admissionRouteDiagnostics, error) {
	cctx, cancel := context.WithTimeout(ctx, chorusTimeout)
	defer cancel()
	cells := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_CELLS", 4, 1, 8))
	frag := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_FRAG", 8, 2, 32))
	args := []string{model, prompt, "field", cells, frag, "1", "0", "0", "0.3", "1", "1.3", "0", "1", strconv.Itoa(qloop), "0"}
	cmd := exec.CommandContext(cctx, bin, args...)
	cmd.Env = append(os.Environ(), "A2A_FIELD_PROMPT_FORMAT=raw")
	out, err := cmd.CombinedOutput()
	if err != nil {
		if cctx.Err() != nil {
			return nil, admissionRouteDiagnostics{}, cctx.Err()
		}
		return nil, admissionRouteDiagnostics{}, fmt.Errorf("%w: %s", err, strings.TrimSpace(strings.ToValidUTF8(string(out), "")))
	}
	raw := string(out)
	diag := parseAdmissionRouteDiagnostics(raw)
	if qloop <= 0 {
		diag.QloopRoutes = 0
		diag.QloopQSrc = 0
		diag.QloopSSrc = 0
		diag.QloopScoreDrop = 0
		diag.QloopPickerSeen = false
	}
	return parseChorusCells(raw), diag, nil
}

func parseAdmissionDirectOutput(out, prompt string) string {
	parts := strings.Split(out, "\n---\n")
	if len(parts) >= 3 {
		out = parts[1]
	}
	out = strings.TrimSpace(out)
	if prompt != "" && strings.HasPrefix(out, prompt) {
		out = strings.TrimSpace(strings.TrimPrefix(out, prompt))
	}
	return cleanRouteText(out)
}

func cleanRouteText(s string) string {
	s = strings.ToValidUTF8(s, "")
	s = stripLabel(strings.Join(strings.Fields(s), " "))
	return cutSentence(s)
}

func filterChorusCells(cells []chorusCell, qloop bool) []chorusCell {
	out := make([]chorusCell, 0, len(cells))
	for _, c := range cells {
		if c.qloop == qloop {
			out = append(out, c)
		}
	}
	return out
}

var routeTimingRe = regexp.MustCompile(`timing: base_ms=\S+ base_gen=(\d+) base_retry=(\d+) base_probe=(\d+) base_rescue=(\d+) base_fail=(\d+) qloop_ms=\S+ qloop_gen=(\d+) qloop_retry=(\d+)(?: qloop_routes=(\d+) qloop_qsrc=(\d+) qloop_ssrc=(\d+) qloop_score_reject=(\d+))?`)

func parseAdmissionRouteDiagnostics(out string) admissionRouteDiagnostics {
	diag := admissionRouteDiagnostics{
		QloopGates:       strings.Count(out, "↳ qloop gate "),
		QloopGateSurface: countQloopGateReason(out, "surface"),
		QloopGateIQ:      countQloopGateReason(out, "iq"),
	}
	m := routeTimingRe.FindStringSubmatch(out)
	if len(m) >= 8 {
		diag.TimingSeen = true
		diag.BaseGenerated = atoiZero(m[1])
		diag.BaseRetries = atoiZero(m[2])
		diag.BaseProbe = atoiZero(m[3])
		diag.BaseRescue = atoiZero(m[4])
		diag.BaseFailed = atoiZero(m[5])
		diag.QloopGenerated = atoiZero(m[6])
		diag.QloopRetries = atoiZero(m[7])
		if len(m) >= 12 && m[8] != "" {
			diag.QloopPickerSeen = true
			diag.QloopRoutes = atoiZero(m[8])
			diag.QloopQSrc = atoiZero(m[9])
			diag.QloopSSrc = atoiZero(m[10])
			diag.QloopScoreDrop = atoiZero(m[11])
		}
	}
	return diag
}

func countQloopGateReason(out, reason string) int {
	n := 0
	needle := "reason=" + reason
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "qloop gate") && strings.Contains(line, needle) {
			n++
		}
	}
	return n
}

func routeEmptyHint(route string, diag admissionRouteDiagnostics) string {
	if route != "qloop" {
		return "empty generation"
	}
	if diag.TimingSeen {
		reason := fmt.Sprintf("no qloop candidate lines (qloop_gen=%d qloop_retry=%d qloop_gates=%d)", diag.QloopGenerated, diag.QloopRetries, diag.QloopGates)
		if diag.QloopPickerSeen {
			reason = fmt.Sprintf("%s routes=%d qsrc=%d ssrc=%d score_drop=%d",
				reason, diag.QloopRoutes, diag.QloopQSrc, diag.QloopSSrc, diag.QloopScoreDrop)
		}
		return reason
	}
	if diag.QloopGates > 0 {
		return fmt.Sprintf("only rejected qloop gates (qloop_gates=%d)", diag.QloopGates)
	}
	return "no qloop candidate lines"
}

func atoiZero(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

func recordAdmissionRouteCandidate(iw *InnerWorld, summary *admissionRouteCompareSummary, index int, out admissionRouteOutput, trigger, seed, fragment string) error {
	st := summary.ByRoute[out.route]
	st.Attempted++
	st.ChorusVoices += out.voices
	st.QloopQuestions += out.questions
	st.QloopGates += out.diag.QloopGates
	st.QloopGateSurface += out.diag.QloopGateSurface
	st.QloopGateIQ += out.diag.QloopGateIQ
	st.QloopGenerated += out.diag.QloopGenerated
	st.QloopRetries += out.diag.QloopRetries
	st.QloopRoutes += out.diag.QloopRoutes
	st.QloopQSrc += out.diag.QloopQSrc
	st.QloopSSrc += out.diag.QloopSSrc
	st.QloopScoreDrop += out.diag.QloopScoreDrop
	if out.diag.QloopPickerSeen {
		st.QloopPickerSeen++
	}
	st.BaseGenerated += out.diag.BaseGenerated
	st.BaseRetries += out.diag.BaseRetries
	st.BaseProbe += out.diag.BaseProbe
	st.BaseRescue += out.diag.BaseRescue
	st.BaseFailed += out.diag.BaseFailed
	if out.diag.TimingSeen {
		st.TimingSeen++
	}
	text := strings.TrimSpace(out.text)
	if text == "" {
		st.Empty++
		summary.EmptyCandidates++
		reason := out.emptyHint
		if reason == "" {
			reason = "empty generation"
		}
		summary.Empties = append(summary.Empties, admissionRouteEmpty{Index: index, Route: out.route, Trigger: trigger, Seed: seed, Reason: reason})
		summary.ByRoute[out.route] = st
		return nil
	}
	st.Produced++
	summary.Candidates++

	before := iw.GetSnapshot()
	r := dreamResult{
		frag:      fragment,
		dream:     text,
		candidate: newDreamCandidate(out.route, trigger, seed, fragment, text, out.cells),
	}
	if admitDreamToInnerWorld(iw, &r, trigger) {
		return fmt.Errorf("sample %d route %s was admitted in shadow mode", index, out.route)
	}
	after := iw.GetSnapshot()
	if after != before {
		return fmt.Errorf("sample %d route %s mutated live inner-world: before=%+v after=%+v", index, out.route, before, after)
	}
	if strings.HasPrefix(r.candidate.Reason, "admission log failed:") {
		return fmt.Errorf("sample %d route %s: %s", index, out.route, r.candidate.Reason)
	}
	if counterfactualReplayOK(r.candidate.Counterfactual) {
		// Expected path.
	} else {
		st.ReplayFailed++
		summary.ReplayFailed++
		replay := "missing replay"
		if r.candidate.Counterfactual != nil && r.candidate.Counterfactual.Replay != nil && r.candidate.Counterfactual.Replay.Reason != "" {
			replay = r.candidate.Counterfactual.Replay.Reason
		}
		summary.Failures = append(summary.Failures, admissionRouteFailure{
			Index:   index,
			Route:   out.route,
			RunID:   r.candidate.RunID,
			Source:  r.candidate.Source,
			Trigger: r.candidate.Trigger,
			Seed:    r.candidate.Seed,
			Replay:  replay,
		})
	}
	if r.candidate.Admission != nil && r.candidate.Admission.Passed {
		st.PolicyPassed++
		summary.PolicyPassed++
	} else {
		st.PolicyFailed++
		summary.PolicyFailed++
		var reasons []string
		if r.candidate.Admission != nil {
			reasons = append([]string(nil), r.candidate.Admission.Reasons...)
		}
		summary.Failures = append(summary.Failures, admissionRouteFailure{
			Index:   index,
			Route:   out.route,
			RunID:   r.candidate.RunID,
			Source:  r.candidate.Source,
			Trigger: r.candidate.Trigger,
			Seed:    r.candidate.Seed,
			Reasons: reasons,
		})
	}
	summary.ByRoute[out.route] = st
	return nil
}

func writeAdmissionRouteCompareSummary(path string, summary admissionRouteCompareSummary) error {
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

func envIntClamped(name string, def, lo, hi int) int {
	if hi < lo {
		hi = lo
	}
	v := def
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil {
			v = n
		}
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func envString(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}
