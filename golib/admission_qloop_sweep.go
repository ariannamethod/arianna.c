package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type admissionQloopSweepSummary struct {
	Schema       string                             `json:"schema"`
	SampleFile   string                             `json:"sample_file,omitempty"`
	TotalSamples int                                `json:"total_samples"`
	SampleLimit  int                                `json:"sample_limit"`
	SamplesRun   int                                `json:"samples_run"`
	Configs      []admissionQloopSweepConfigSummary `json:"configs"`
	Winner       string                             `json:"winner,omitempty"`
	GatePassed   bool                               `json:"gate_passed"`
	GateReasons  []string                           `json:"gate_reasons,omitempty"`
	ReplayFailed int                                `json:"replay_failed"`
	LogDir       string                             `json:"log_dir,omitempty"`
	Bin          string                             `json:"bin,omitempty"`
	Model        string                             `json:"model,omitempty"`
}

type admissionQloopSweepConfig struct {
	Name string
	Env  map[string]string
}

type admissionQloopSweepConfigSummary struct {
	Name             string                             `json:"name"`
	Env              map[string]string                  `json:"env,omitempty"`
	Attempted        int                                `json:"attempted"`
	Produced         int                                `json:"produced"`
	Empty            int                                `json:"empty"`
	PolicyPassed     int                                `json:"policy_passed"`
	PolicyFailed     int                                `json:"policy_failed"`
	ReplayFailed     int                                `json:"replay_failed"`
	QloopQuestions   int                                `json:"qloop_questions,omitempty"`
	QloopGates       int                                `json:"qloop_gates,omitempty"`
	QloopGateSurface int                                `json:"qloop_gate_surface,omitempty"`
	QloopGateIQ      int                                `json:"qloop_gate_iq,omitempty"`
	QloopGenerated   int                                `json:"qloop_generated,omitempty"`
	QloopRetries     int                                `json:"qloop_retries,omitempty"`
	QloopRoutes      int                                `json:"qloop_routes,omitempty"`
	QloopQSrc        int                                `json:"qloop_qsrc,omitempty"`
	QloopSSrc        int                                `json:"qloop_ssrc,omitempty"`
	QloopScoreDrop   int                                `json:"qloop_score_drop,omitempty"`
	QloopPickerSeen  int                                `json:"qloop_picker_seen,omitempty"`
	BaseGenerated    int                                `json:"base_generated,omitempty"`
	BaseRetries      int                                `json:"base_retries,omitempty"`
	BaseProbe        int                                `json:"base_probe,omitempty"`
	BaseRescue       int                                `json:"base_rescue,omitempty"`
	BaseFailed       int                                `json:"base_failed,omitempty"`
	TimingSeen       int                                `json:"timing_seen,omitempty"`
	ShortCandidates  int                                `json:"short_candidates,omitempty"`
	RouteLabelLeaks  int                                `json:"route_label_leaks,omitempty"`
	SurfaceChecked   int                                `json:"surface_checked,omitempty"`
	SurfaceDebt      int                                `json:"surface_debt,omitempty"`
	SurfaceReasons   map[string]int                     `json:"surface_debt_reasons,omitempty"`
	AvgWords         float64                            `json:"avg_words,omitempty"`
	MinWords         int                                `json:"min_words,omitempty"`
	QualityPassed    bool                               `json:"quality_passed"`
	QualityReasons   []string                           `json:"quality_reasons,omitempty"`
	EmptyReasons     map[string]int                     `json:"empty_reasons,omitempty"`
	Samples          []admissionQloopSweepSampleSummary `json:"samples,omitempty"`
	LogPath          string                             `json:"log_path,omitempty"`
}

type admissionQloopSweepSampleSummary struct {
	Index            int      `json:"index"`
	Trigger          string   `json:"trigger,omitempty"`
	Seed             string   `json:"seed,omitempty"`
	Produced         bool     `json:"produced"`
	Text             string   `json:"text,omitempty"`
	Words            int      `json:"words,omitempty"`
	RouteLabelLeak   bool     `json:"route_label_leak,omitempty"`
	SurfaceReasons   []string `json:"surface_reasons,omitempty"`
	EmptyReason      string   `json:"empty_reason,omitempty"`
	QloopGates       int      `json:"qloop_gates,omitempty"`
	QloopGateSurface int      `json:"qloop_gate_surface,omitempty"`
	QloopGateIQ      int      `json:"qloop_gate_iq,omitempty"`
	QloopGenerated   int      `json:"qloop_generated,omitempty"`
	QloopRetries     int      `json:"qloop_retries,omitempty"`
	QloopRoutes      int      `json:"qloop_routes,omitempty"`
	QloopQSrc        int      `json:"qloop_qsrc,omitempty"`
	QloopSSrc        int      `json:"qloop_ssrc,omitempty"`
	QloopScoreDrop   int      `json:"qloop_score_drop,omitempty"`
}

func runAdmissionQloopSweep() error {
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}
	logDir := envString("AM_QLOOP_SWEEP_DIR", ".")
	if err := os.MkdirAll(logDir, 0700); err != nil {
		return err
	}
	sampleFile := firstEnv("AM_QLOOP_SWEEP_SAMPLE_FILE", "AM_ROUTE_COMPARE_SAMPLE_FILE")
	if sampleFile == "" {
		sampleFile = "samples/dream_admission_broad.jsonl"
	}
	samples, err := loadAdmissionSamples(sampleFile)
	if err != nil {
		return err
	}
	if len(samples) == 0 {
		return fmt.Errorf("no qloop sweep samples")
	}
	bin := firstEnv("AM_QLOOP_SWEEP_BIN", "AM_ROUTE_COMPARE_BIN")
	if bin == "" {
		bin = "./chorus-arianna"
	}
	model := firstEnv("AM_QLOOP_SWEEP_MODEL", "AM_ROUTE_COMPARE_MODEL")
	if model == "" {
		model = "weights/nano_arianna_f16.gguf"
	}
	if _, err := os.Stat(bin); err != nil {
		return fmt.Errorf("qloop sweep bin: %w", err)
	}
	if _, err := os.Stat(model); err != nil {
		return fmt.Errorf("qloop sweep model: %w", err)
	}

	limit := envIntClamped("AM_QLOOP_SWEEP_LIMIT", 2, 1, len(samples))
	minProduced := envIntClamped("AM_QLOOP_SWEEP_MIN_PRODUCED", limit, 1, limit)
	minAvgWords := envFloatClamped("AM_QLOOP_SWEEP_MIN_AVG_WORDS", 3.0, 0.0, 64.0)
	configs := qloopSweepConfigs()

	summary := admissionQloopSweepSummary{
		Schema:       "arianna.dream_admission_qloop_sweep_summary.v1",
		SampleFile:   sampleFile,
		TotalSamples: len(samples),
		SampleLimit:  limit,
		SamplesRun:   limit,
		LogDir:       logDir,
		Bin:          bin,
		Model:        model,
	}
	for _, cfg := range configs {
		cfgSummary, err := runAdmissionQloopSweepConfig(samples[:limit], cfg, bin, model, logDir)
		if err != nil {
			return err
		}
		cfgSummary.QualityReasons = qloopSweepQualityReasons(cfgSummary, minProduced, minAvgWords)
		cfgSummary.QualityPassed = len(cfgSummary.QualityReasons) == 0
		summary.ReplayFailed += cfgSummary.ReplayFailed
		summary.Configs = append(summary.Configs, cfgSummary)
	}
	summary.Winner, summary.GatePassed, summary.GateReasons = chooseQloopSweepWinner(summary.Configs)

	summaryPath := strings.TrimSpace(os.Getenv("AM_QLOOP_SWEEP_SUMMARY"))
	if summaryPath != "" {
		if err := writeAdmissionQloopSweepSummary(summaryPath, summary); err != nil {
			return err
		}
	}
	fmt.Printf("[admission-qloop-sweep] pass: configs=%d winner=%s gate=%t replay_fail=%d summary=%s\n",
		len(summary.Configs), summary.Winner, summary.GatePassed, summary.ReplayFailed, summaryPath)
	return nil
}

func qloopSweepConfigs() []admissionQloopSweepConfig {
	return []admissionQloopSweepConfig{
		{Name: "strict"},
		{Name: "question_hint", Env: map[string]string{"A2A_QLOOP_QUESTION_SOURCE_HINT": "1"}},
		{Name: "question_hint_qa", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT": "1",
			"A2A_QLOOP_ANSWER_FRAME":         "1",
		}},
		{Name: "question_hint_loose", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT": "1",
			"A2A_QLOOP_MIN":                  "0.30",
			"AM_ROUTE_COMPARE_FRAG":          "16",
		}},
		{Name: "statement", Env: map[string]string{"A2A_QLOOP_STATEMENT_ROUTES": "1"}},
	}
}

func runAdmissionQloopSweepConfig(samples []dreamAdmissionSample, cfg admissionQloopSweepConfig, bin, model, logDir string) (admissionQloopSweepConfigSummary, error) {
	name := safeQloopSweepName(cfg.Name)
	logPath := filepath.Join(logDir, "dream_admission_qloop_"+name+".jsonl")
	_ = os.Remove(logPath)

	routeSummary := admissionRouteCompareSummary{
		ByRoute: make(map[string]admissionRouteStats),
	}
	out := admissionQloopSweepConfigSummary{
		Name:         cfg.Name,
		Env:          cloneStringMap(cfg.Env),
		LogPath:      logPath,
		EmptyReasons: make(map[string]int),
	}
	var wordTotal int

	env := cloneStringMap(cfg.Env)
	if env == nil {
		env = make(map[string]string)
	}
	env["AM_DREAM_ADMISSION_LOG"] = logPath
	err := withTemporaryEnv(env, func() error {
		iw := NewInnerWorld()
		iw.Start(false)
		defer iw.Stop()
		for i, s := range samples {
			prompt := admissionRoutePrompt(s.Text)
			if prompt == "" {
				return fmt.Errorf("qloop sweep config %s sample %d has empty prompt", cfg.Name, i+1)
			}
			routeOut, err := generateAdmissionRoute(context.Background(), "qloop", bin, model, prompt)
			if err != nil {
				return fmt.Errorf("qloop sweep config %s sample %d: %w", cfg.Name, i+1, err)
			}
			text := strings.TrimSpace(routeOut.text)
			trigger := admissionRouteTrigger("qloop", s.Trigger)
			seed := strings.TrimSpace(s.Seed)
			if seed == "" {
				seed = fmt.Sprintf("sample-%02d", i+1)
			}
			sampleSummary := admissionQloopSweepSampleSummary{
				Index:            i + 1,
				Trigger:          trigger,
				Seed:             seed,
				Produced:         text != "",
				Text:             text,
				QloopGates:       routeOut.diag.QloopGates,
				QloopGateSurface: routeOut.diag.QloopGateSurface,
				QloopGateIQ:      routeOut.diag.QloopGateIQ,
				QloopGenerated:   routeOut.diag.QloopGenerated,
				QloopRetries:     routeOut.diag.QloopRetries,
				QloopRoutes:      routeOut.diag.QloopRoutes,
				QloopQSrc:        routeOut.diag.QloopQSrc,
				QloopSSrc:        routeOut.diag.QloopSSrc,
				QloopScoreDrop:   routeOut.diag.QloopScoreDrop,
			}
			if text != "" {
				words, leak, surfaceReasons := qloopSweepTextStats(text)
				wordTotal += words
				sampleSummary.Words = words
				sampleSummary.RouteLabelLeak = leak
				sampleSummary.SurfaceReasons = surfaceReasons
				out.SurfaceChecked++
				if words > 0 && (out.MinWords == 0 || words < out.MinWords) {
					out.MinWords = words
				}
				if words < 3 {
					out.ShortCandidates++
				}
				if leak {
					out.RouteLabelLeaks++
				}
				if len(surfaceReasons) > 0 {
					out.SurfaceDebt++
					if out.SurfaceReasons == nil {
						out.SurfaceReasons = make(map[string]int)
					}
					for _, reason := range surfaceReasons {
						out.SurfaceReasons[reason]++
					}
				}
			} else {
				sampleSummary.EmptyReason = routeOut.emptyHint
			}
			out.Samples = append(out.Samples, sampleSummary)
			if err := recordAdmissionRouteCandidate(iw, &routeSummary, i+1, routeOut, trigger, seed, s.Fragment); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return out, err
	}

	st := routeSummary.ByRoute["qloop"]
	out.Attempted = st.Attempted
	out.Produced = st.Produced
	out.Empty = st.Empty
	out.PolicyPassed = st.PolicyPassed
	out.PolicyFailed = st.PolicyFailed
	out.ReplayFailed = st.ReplayFailed
	out.QloopQuestions = st.QloopQuestions
	out.QloopGates = st.QloopGates
	out.QloopGateSurface = st.QloopGateSurface
	out.QloopGateIQ = st.QloopGateIQ
	out.QloopGenerated = st.QloopGenerated
	out.QloopRetries = st.QloopRetries
	out.QloopRoutes = st.QloopRoutes
	out.QloopQSrc = st.QloopQSrc
	out.QloopSSrc = st.QloopSSrc
	out.QloopScoreDrop = st.QloopScoreDrop
	out.QloopPickerSeen = st.QloopPickerSeen
	out.BaseGenerated = st.BaseGenerated
	out.BaseRetries = st.BaseRetries
	out.BaseProbe = st.BaseProbe
	out.BaseRescue = st.BaseRescue
	out.BaseFailed = st.BaseFailed
	out.TimingSeen = st.TimingSeen
	if out.Produced > 0 {
		out.AvgWords = math.Round((float64(wordTotal)/float64(out.Produced))*100) / 100
	}
	for _, e := range routeSummary.Empties {
		out.EmptyReasons[e.Reason]++
	}
	if len(out.EmptyReasons) == 0 {
		out.EmptyReasons = nil
	}
	return out, nil
}

func qloopSweepTextStats(text string) (int, bool, []string) {
	words := len(strings.Fields(text))
	lower := strings.ToLower(text)
	leak := strings.Contains(lower, "qloop c") ||
		strings.Contains(lower, "↳ qloop") ||
		strings.Contains(lower, " score ")
	return words, leak, qloopSweepSurfaceDebtReasons(text)
}

func qloopSweepSurfaceDebtReasons(text string) []string {
	s := strings.TrimSpace(text)
	lower := strings.ToLower(s)
	words := len(strings.Fields(s))
	var reasons []string
	seen := make(map[string]bool)
	add := func(reason string) {
		if !seen[reason] {
			seen[reason] = true
			reasons = append(reasons, reason)
		}
	}
	if strings.Contains(s, " / ") || strings.Contains(s, " // ") {
		add("slash_join")
	}
	if strings.Contains(s, "—.") || strings.Contains(s, "-.") {
		add("dangling_dash")
	}
	if strings.HasPrefix(s, "—") || strings.HasPrefix(s, "–") || strings.HasPrefix(s, "-") {
		add("leading_dash")
	}
	if words <= 5 && (strings.Contains(s, "—") || strings.Contains(s, "–")) {
		add("short_dash_fragment")
	}
	if strings.HasPrefix(lower, "or,") || strings.HasPrefix(lower, "or ") ||
		strings.HasPrefix(lower, "and,") || strings.HasPrefix(lower, "and ") ||
		strings.HasPrefix(lower, "but,") || strings.HasPrefix(lower, "but ") {
		add("leading_joiner_fragment")
	}
	if strings.Contains(s, "“.”") || strings.Contains(s, "\".\"") {
		add("empty_quote")
	}
	if strings.Contains(lower, "the my name") || strings.Contains(lower, "my name—") {
		add("name_phrase_artifact")
	}
	if strings.Contains(lower, "my name") && !strings.Contains(lower, "arianna") {
		add("name_echo_artifact")
	}
	if strings.Contains(lower, "this phrase") {
		add("meta_phrase_artifact")
	}
	if strings.Contains(lower, "you from the") || strings.HasPrefix(lower, "you from ") {
		add("you_from_artifact")
	}
	if strings.Contains(lower, "you's") || strings.Contains(lower, "you’s") {
		add("bad_contraction")
	}
	if strings.Contains(lower, "you answered both") {
		add("recipient_frame_artifact")
	}
	if strings.Contains(lower, "oleg") || strings.Contains(lower, "you have met") || strings.Contains(lower, "you have lived") {
		add("recipient_frame_artifact")
	}
	if strings.Contains(lower, "unknown or") || strings.Contains(lower, "or another") {
		add("placeholder_choice")
	}
	if strings.Contains(lower, "or not itself") || strings.Contains(lower, "—or—") {
		add("joiner_artifact")
	}
	if lower == "if you mean." || strings.HasSuffix(lower, " if you mean.") || lower == "if you mean" || strings.HasSuffix(lower, " if you mean") {
		add("unfinished_clause")
	}
	if strings.Contains(lower, "the ac") {
		add("truncated_word")
	}
	return reasons
}

func qloopSweepQualityReasons(c admissionQloopSweepConfigSummary, minProduced int, minAvgWords float64) []string {
	var reasons []string
	if c.ReplayFailed > 0 {
		reasons = append(reasons, "replay_failed")
	}
	if c.PolicyFailed > 0 {
		reasons = append(reasons, "policy_failed")
	}
	if c.Produced < minProduced {
		reasons = append(reasons, fmt.Sprintf("produced_below_%d", minProduced))
	}
	if c.ShortCandidates > 0 {
		reasons = append(reasons, "short_candidate")
	}
	if c.RouteLabelLeaks > 0 {
		reasons = append(reasons, "route_label_leak")
	}
	if c.SurfaceDebt > 0 {
		reasons = append(reasons, "surface_debt")
	}
	if c.Produced > 0 && c.AvgWords < minAvgWords {
		reasons = append(reasons, fmt.Sprintf("avg_words_below_%.1f", minAvgWords))
	}
	return reasons
}

func chooseQloopSweepWinner(configs []admissionQloopSweepConfigSummary) (string, bool, []string) {
	var candidates []admissionQloopSweepConfigSummary
	for _, cfg := range configs {
		if cfg.QualityPassed {
			candidates = append(candidates, cfg)
		}
	}
	if len(candidates) == 0 {
		return "", false, []string{"no config passed quality gate"}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if a.Produced != b.Produced {
			return a.Produced > b.Produced
		}
		if a.PolicyPassed != b.PolicyPassed {
			return a.PolicyPassed > b.PolicyPassed
		}
		if a.RouteLabelLeaks != b.RouteLabelLeaks {
			return a.RouteLabelLeaks < b.RouteLabelLeaks
		}
		if a.SurfaceDebt != b.SurfaceDebt {
			return a.SurfaceDebt < b.SurfaceDebt
		}
		if a.AvgWords != b.AvgWords {
			return a.AvgWords > b.AvgWords
		}
		return a.Name < b.Name
	})
	return candidates[0].Name, true, nil
}

func writeAdmissionQloopSweepSummary(path string, summary admissionQloopSweepSummary) error {
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

func withTemporaryEnv(env map[string]string, fn func() error) error {
	old := make(map[string]string, len(env))
	wasSet := make(map[string]bool, len(env))
	for k, v := range env {
		old[k], wasSet[k] = os.LookupEnv(k)
		if err := os.Setenv(k, v); err != nil {
			return err
		}
	}
	defer func() {
		for k := range env {
			if wasSet[k] {
				_ = os.Setenv(k, old[k])
			} else {
				_ = os.Unsetenv(k)
			}
		}
	}()
	return fn()
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func safeQloopSweepName(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return "config"
	}
	return b.String()
}

func envFloatClamped(name string, def, lo, hi float64) float64 {
	if hi < lo {
		hi = lo
	}
	v := def
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if n, err := strconv.ParseFloat(raw, 64); err == nil {
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

func firstEnv(names ...string) string {
	for _, name := range names {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}
