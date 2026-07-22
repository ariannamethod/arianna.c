package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type admissionRouteCompareSummary struct {
	Schema                  string                           `json:"schema"`
	SampleFile              string                           `json:"sample_file,omitempty"`
	TotalSamples            int                              `json:"total_samples"`
	SampleLimit             int                              `json:"sample_limit"`
	SamplesRun              int                              `json:"samples_run"`
	Routes                  []string                         `json:"routes"`
	Candidates              int                              `json:"candidates"`
	EmptyCandidates         int                              `json:"empty_candidates"`
	PolicyPassed            int                              `json:"policy_passed"`
	PolicyFailed            int                              `json:"policy_failed"`
	ReplayFailed            int                              `json:"replay_failed"`
	SemanticPassed          int                              `json:"semantic_passed"`
	SemanticMiss            int                              `json:"semantic_miss"`
	SemanticScore           int                              `json:"semantic_score"`
	SemanticCoveragePassed  bool                             `json:"semantic_coverage_passed"`
	SemanticCoverageReasons []string                         `json:"semantic_coverage_reasons,omitempty"`
	SemanticAdmission       admissionRouteSemanticAdmission  `json:"semantic_route_admission"`
	ShadowBestRoute         admissionRouteShadowBestRoute    `json:"shadow_best_route"`
	ByRoute                 map[string]admissionRouteStats   `json:"by_route"`
	Failures                []admissionRouteFailure          `json:"failures,omitempty"`
	Empties                 []admissionRouteEmpty            `json:"empties,omitempty"`
	SemanticSamples         []admissionRouteSemantic         `json:"semantic_samples,omitempty"`
	SemanticCoverage        []admissionRouteSemanticCoverage `json:"semantic_coverage,omitempty"`
	LogPath                 string                           `json:"log_path,omitempty"`
	Bin                     string                           `json:"bin,omitempty"`
	Model                   string                           `json:"model,omitempty"`
}

type admissionRouteStats struct {
	Attempted        int `json:"attempted"`
	Produced         int `json:"produced"`
	Empty            int `json:"empty"`
	PolicyPassed     int `json:"policy_passed"`
	PolicyFailed     int `json:"policy_failed"`
	ReplayFailed     int `json:"replay_failed"`
	SemanticPassed   int `json:"semantic_passed"`
	SemanticMiss     int `json:"semantic_miss"`
	SemanticScore    int `json:"semantic_score"`
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
	QloopTypedSrc    int `json:"qloop_tsrc,omitempty"`
	QloopTargetCtx   int `json:"qloop_tctx,omitempty"`
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

type admissionRouteSemantic struct {
	Index       int      `json:"index"`
	Route       string   `json:"route"`
	Trigger     string   `json:"trigger"`
	Seed        string   `json:"seed"`
	PromptClass string   `json:"prompt_class"`
	Text        string   `json:"text"`
	Score       int      `json:"score"`
	Passed      bool     `json:"passed"`
	Reasons     []string `json:"reasons,omitempty"`
}

type admissionRouteSemanticCoverage struct {
	Index          int      `json:"index"`
	Seed           string   `json:"seed"`
	PromptClass    string   `json:"prompt_class"`
	Attempted      int      `json:"attempted"`
	Produced       int      `json:"produced"`
	Empty          int      `json:"empty"`
	SemanticPassed int      `json:"semantic_passed"`
	SemanticMiss   int      `json:"semantic_miss"`
	BestRoute      string   `json:"best_route,omitempty"`
	BestText       string   `json:"best_text,omitempty"`
	BestScore      *int     `json:"best_score,omitempty"`
	BestPassed     bool     `json:"best_passed,omitempty"`
	BestReasons    []string `json:"best_reasons,omitempty"`
}

type admissionRouteSemanticAdmission struct {
	Passed       bool                                    `json:"passed"`
	Reasons      []string                                `json:"reasons,omitempty"`
	Reviews      int                                     `json:"reviews"`
	Admitted     int                                     `json:"admitted"`
	Rejected     int                                     `json:"rejected"`
	NoCandidate  int                                     `json:"no_candidate,omitempty"`
	SemanticMiss int                                     `json:"semantic_miss,omitempty"`
	Decisions    []admissionRouteSemanticAdmissionReview `json:"decisions,omitempty"`
}

type admissionRouteSemanticAdmissionReview struct {
	Index           int      `json:"index"`
	Seed            string   `json:"seed"`
	PromptClass     string   `json:"prompt_class"`
	Decision        string   `json:"decision"`
	Reason          string   `json:"reason,omitempty"`
	Route           string   `json:"route,omitempty"`
	Text            string   `json:"text,omitempty"`
	Score           *int     `json:"score,omitempty"`
	SemanticReasons []string `json:"semantic_reasons,omitempty"`
	CandidatesSeen  int      `json:"candidates_seen"`
	EmptyRoutes     int      `json:"empty_routes,omitempty"`
	AttemptedRoutes int      `json:"attempted_routes"`
}

type admissionRouteShadowBestRoute struct {
	Schema        string                               `json:"schema"`
	Passed        bool                                 `json:"passed"`
	Reviews       int                                  `json:"reviews"`
	Selected      int                                  `json:"selected"`
	Rejected      int                                  `json:"rejected,omitempty"`
	Reasons       []string                             `json:"reasons,omitempty"`
	ByRoute       map[string]admissionRouteShadowStats `json:"by_route,omitempty"`
	RoutePlan     []admissionRouteShadowDecision       `json:"route_plan,omitempty"`
	Rejects       []admissionRouteShadowDecision       `json:"rejects,omitempty"`
	SemanticScore int                                  `json:"semantic_score"`
}

type admissionRouteShadowStats struct {
	Selected      int      `json:"selected"`
	SemanticScore int      `json:"semantic_score"`
	PromptClasses []string `json:"prompt_classes,omitempty"`
}

type admissionRouteShadowDecision struct {
	Index           int      `json:"index"`
	Seed            string   `json:"seed"`
	PromptClass     string   `json:"prompt_class"`
	Route           string   `json:"route,omitempty"`
	Text            string   `json:"text,omitempty"`
	Score           *int     `json:"score,omitempty"`
	Reason          string   `json:"reason,omitempty"`
	SemanticReasons []string `json:"semantic_reasons,omitempty"`
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
	QloopTypedSrc    int
	QloopTargetCtx   int
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
	progress := admissionRouteCompareProgressWriter()

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
		seed := strings.TrimSpace(s.Seed)
		if seed == "" {
			seed = fmt.Sprintf("sample-%02d", i+1)
		}
		promptClass := qloopSweepPromptClass(s.Trigger, seed)
		summary.SamplesRun++
		admissionRouteProgressf(progress, "sample=%d/%d seed=%s class=%s routes=%d", i+1, limit, seed, promptClass, len(routes))
		for ri, route := range routes {
			admissionRouteProgressf(progress, "sample=%d/%d route=%d/%d route=%s start", i+1, limit, ri+1, len(routes), route)
			out, err := generateAdmissionRouteWithPromptClass(context.Background(), route, bin, model, prompt, promptClass)
			if err != nil {
				return fmt.Errorf("sample %d route %s: %w", i+1, route, err)
			}
			trigger := admissionRouteTrigger(route, s.Trigger)
			admissionRouteProgressOutput(progress, i+1, limit, ri+1, len(routes), route, trigger, seed, promptClass, out)
			if err := recordAdmissionRouteCandidate(iw, &summary, i+1, out, trigger, seed, s.Fragment); err != nil {
				return err
			}
		}
	}
	summary.SemanticCoverage = buildAdmissionRouteSemanticCoverage(summary.SemanticSamples, summary.Empties)
	summary.SemanticCoveragePassed, summary.SemanticCoverageReasons = summarizeAdmissionRouteSemanticCoverage(summary.SemanticCoverage)
	summary.SemanticAdmission = buildAdmissionRouteSemanticAdmission(summary.SemanticCoverage)
	summary.ShadowBestRoute = buildAdmissionRouteShadowBestRoute(summary.SemanticAdmission)

	summaryPath := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_SUMMARY"))
	if summaryPath != "" {
		if err := writeAdmissionRouteCompareSummary(summaryPath, summary); err != nil {
			return err
		}
	}
	fmt.Println(formatAdmissionRouteShadowBestRouteLine(summary.ShadowBestRoute))
	fmt.Printf("[admission-route-compare] pass: samples=%d/%d candidates=%d empty=%d policy_fail=%d replay_fail=%d log=%s summary=%s\n",
		summary.SamplesRun, summary.TotalSamples, summary.Candidates, summary.EmptyCandidates, summary.PolicyFailed, summary.ReplayFailed, logPath, summaryPath)
	return nil
}

func admissionRouteCompareProgressWriter() io.Writer {
	if !admissionRouteCompareProgressEnabled() {
		return nil
	}
	return os.Stderr
}

func admissionRouteCompareProgressEnabled() bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_PROGRESS")))
	switch raw {
	case "", "1", "true", "yes", "on", "stderr":
		return true
	case "0", "false", "no", "off", "none":
		return false
	default:
		return true
	}
}

func admissionRouteProgressf(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	fmt.Fprintf(w, "[admission-route-compare] progress "+format+"\n", args...)
}

func admissionRouteProgressOutput(w io.Writer, sample, total, routeIndex, routeTotal int, route, trigger, seed, promptClass string, out admissionRouteOutput) {
	if w == nil {
		return
	}
	text := strings.TrimSpace(out.text)
	if text == "" {
		reason := out.emptyHint
		if reason == "" {
			reason = "empty generation"
		}
		admissionRouteProgressf(w, "sample=%d/%d route=%d/%d route=%s done empty reason=%q", sample, total, routeIndex, routeTotal, route, reason)
		return
	}
	semantic := qloopSweepSemanticAssessment(text, qloopSweepPromptClass(trigger, seed))
	admissionRouteProgressf(w, "sample=%d/%d route=%d/%d route=%s done produced score=%d passed=%t class=%s words=%d", sample, total, routeIndex, routeTotal, route, semantic.Score, semantic.Passed, promptClass, len(strings.Fields(text)))
}

func admissionCompareRoutes() []string {
	raw := strings.TrimSpace(os.Getenv("AM_ROUTE_COMPARE_ROUTES"))
	if raw == "" {
		return []string{"direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"}
	}
	var routes []string
	for _, part := range strings.Split(raw, ",") {
		route := strings.ToLower(strings.TrimSpace(part))
		switch route {
		case "direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge":
			routes = append(routes, route)
		}
	}
	if len(routes) == 0 {
		return []string{"direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"}
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
	return generateAdmissionRouteWithPromptClass(ctx, route, bin, model, prompt, "")
}

func generateAdmissionRouteWithPromptClass(ctx context.Context, route, bin, model, prompt, promptClass string) (admissionRouteOutput, error) {
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
		return generateAdmissionQloopRoute(ctx, route, bin, model, prompt, promptClass, nil)
	case "qloop_hint_qa":
		return generateAdmissionQloopRoute(ctx, route, bin, model, prompt, promptClass, admissionQloopHintQAEnv())
	case "qloop_target":
		return generateAdmissionQloopRoute(ctx, route, bin, model, prompt, promptClass, admissionQloopTargetEnv())
	case "user_bridge":
		return generateAdmissionUserBridge(ctx, bin, model, prompt, promptClass)
	default:
		return admissionRouteOutput{}, fmt.Errorf("unknown route %q", route)
	}
}

func admissionQloopTargetEnv() map[string]string {
	return map[string]string{
		"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
		"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
		"A2A_QLOOP_SOURCE_CLASS":          "prompt",
		"A2A_QLOOP_TARGET_CLASS_HINT":     "1",
	}
}

func admissionQloopHintQAEnv() map[string]string {
	return map[string]string{
		"A2A_QLOOP_QUESTION_SOURCE_HINT": "1",
		"A2A_QLOOP_ANSWER_FRAME":         "1",
	}
}

func generateAdmissionQloopRoute(ctx context.Context, route, bin, model, prompt, promptClass string, env map[string]string) (admissionRouteOutput, error) {
	var cells []chorusCell
	var diag admissionRouteDiagnostics
	run := func() error {
		return withResolvedQloopSourceClass(promptClass, func() error {
			var err error
			cells, diag, err = generateAdmissionChorus(ctx, bin, model, prompt, 2)
			return err
		})
	}
	var err error
	if len(env) > 0 {
		err = withTemporaryEnv(env, run)
	} else {
		err = run()
	}
	if err != nil {
		return admissionRouteOutput{route: route}, err
	}
	cells = filterChorusCells(cells, true)
	voices, questions := chorusCounts(cells)
	return admissionRouteOutput{route: route, text: qloopAdmissionTextForClass(cells, promptClass), cells: cells, voices: voices, questions: questions, diag: diag, emptyHint: routeEmptyHint(route, diag)}, nil
}

func generateAdmissionUserBridge(ctx context.Context, bin, model, prompt, promptClass string) (admissionRouteOutput, error) {
	line := admissionRouteUserLine(prompt)
	if line == "" {
		return admissionRouteOutput{route: "user_bridge"}, fmt.Errorf("empty user bridge prompt")
	}
	cctx, cancel := context.WithTimeout(ctx, chorusTimeout)
	defer cancel()
	cellsArg := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_CELLS", 4, 1, 8))
	baseFrag := envIntClamped("AM_ROUTE_COMPARE_FRAG", 8, 2, 32)
	fragArg := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_USER_FRAG", baseFrag, 2, 64))
	roundsArg := strconv.Itoa(envIntClamped("AM_ROUTE_COMPARE_USER_ROUNDS", 1, 1, 4))
	cmd := exec.CommandContext(cctx, bin, model, "repl", cellsArg, fragArg, roundsArg)
	cmd.Env = append(os.Environ(),
		"A2A_REPL_QLOOP=1",
		"A2A_REPL_PROMPT_FORMAT=user_arianna",
	)
	cmd.Stdin = strings.NewReader(line + "\n:q\n")
	out, err := cmd.CombinedOutput()
	if err != nil {
		if cctx.Err() != nil {
			return admissionRouteOutput{route: "user_bridge"}, cctx.Err()
		}
		return admissionRouteOutput{route: "user_bridge"}, fmt.Errorf("%w: %s", err, strings.TrimSpace(strings.ToValidUTF8(string(out), "")))
	}
	cells := parseChorusCells(string(out))
	voices, questions := chorusCounts(cells)
	return admissionRouteOutput{
		route:     "user_bridge",
		text:      qloopAdmissionTextForClass(cells, promptClass),
		cells:     cells,
		voices:    voices,
		questions: questions,
		emptyHint: "no user bridge candidate lines",
	}, nil
}

func admissionRouteUserLine(prompt string) string {
	s := strings.TrimSpace(prompt)
	lower := strings.ToLower(s)
	if strings.HasPrefix(lower, "q:") {
		s = strings.TrimSpace(s[2:])
		s = stripRouteTrailingLabel(s, "a:")
	}
	lower = strings.ToLower(s)
	if strings.HasPrefix(lower, "user:") {
		s = strings.TrimSpace(s[len("user:"):])
	}
	for _, label := range []string{"arianna:", "assistant:", "a:"} {
		s = stripRouteTrailingLabel(s, label)
	}
	return strings.TrimSpace(s)
}

func stripRouteTrailingLabel(s, label string) string {
	s = strings.TrimSpace(s)
	label = strings.ToLower(label)
	for {
		lower := strings.ToLower(s)
		if !strings.HasSuffix(lower, label) {
			return s
		}
		s = strings.TrimSpace(s[:len(s)-len(label)])
	}
}

func withResolvedQloopSourceClass(promptClass string, fn func() error) error {
	if strings.TrimSpace(os.Getenv("A2A_QLOOP_SOURCE_CLASS")) != "prompt" {
		return fn()
	}
	promptClass = qloopSweepPromptClass(promptClass, promptClass)
	if promptClass == "" || promptClass == "unknown" {
		return fn()
	}
	return withTemporaryEnv(map[string]string{"A2A_QLOOP_SOURCE_CLASS": promptClass}, fn)
}

func qloopAdmissionText(cells []chorusCell) string {
	return qloopAdmissionTextForClass(cells, "")
}

func qloopAdmissionTextForClass(cells []chorusCell, promptClass string) string {
	best := ""
	bestPenalty := 1 << 30
	bestWords := -1
	bestClean := false
	bestSemantic := -1
	for _, cell := range cells {
		text := strings.TrimSpace(cell.text)
		if text == "" {
			continue
		}
		words, penalty, clean := qloopAdmissionCandidateStats(text)
		semantic := 0
		if strings.TrimSpace(promptClass) != "" {
			semantic = qloopSweepSemanticAssessment(text, promptClass).Score
		}
		if best == "" ||
			(clean != bestClean && clean) ||
			(clean == bestClean && semantic != bestSemantic && semantic > bestSemantic) ||
			(clean == bestClean && semantic == bestSemantic && penalty < bestPenalty) ||
			(clean == bestClean && semantic == bestSemantic && penalty == bestPenalty && words > bestWords) {
			best = text
			bestPenalty = penalty
			bestWords = words
			bestClean = clean
			bestSemantic = semantic
		}
	}
	return best
}

func qloopAdmissionCandidateStats(text string) (words, penalty int, clean bool) {
	words, leak, debt := qloopSweepTextStats(text)
	penalty = len(debt) * 20
	if leak {
		penalty += 100
	}
	if words < 3 {
		penalty += 10 + (3 - words)
	}
	return words, penalty, penalty == 0 && words >= 3
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

var routeTimingRe = regexp.MustCompile(`timing: base_ms=\S+ base_gen=(\d+) base_retry=(\d+) base_probe=(\d+) base_rescue=(\d+) base_fail=(\d+) qloop_ms=\S+ qloop_gen=(\d+) qloop_retry=(\d+)(?: qloop_routes=(\d+) qloop_qsrc=(\d+) qloop_ssrc=(\d+) qloop_score_reject=(\d+)(?: qloop_tsrc=(\d+))?(?: qloop_tctx=(\d+))?)?`)

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
			if len(m) >= 13 {
				diag.QloopTypedSrc = atoiZero(m[12])
			}
			if len(m) >= 14 {
				diag.QloopTargetCtx = atoiZero(m[13])
			}
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
	if !isAdmissionQloopRoute(route) {
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

func isAdmissionQloopRoute(route string) bool {
	switch route {
	case "qloop", "qloop_hint_qa", "qloop_target", "user_bridge":
		return true
	default:
		return false
	}
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
	st.QloopTypedSrc += out.diag.QloopTypedSrc
	st.QloopTargetCtx += out.diag.QloopTargetCtx
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
	promptClass := qloopSweepPromptClass(trigger, seed)
	semantic := qloopSweepSemanticAssessment(text, promptClass)
	st.SemanticScore += semantic.Score
	summary.SemanticScore += semantic.Score
	if semantic.Passed {
		st.SemanticPassed++
		summary.SemanticPassed++
	} else {
		st.SemanticMiss++
		summary.SemanticMiss++
	}
	summary.SemanticSamples = append(summary.SemanticSamples, admissionRouteSemantic{
		Index:       index,
		Route:       out.route,
		Trigger:     trigger,
		Seed:        seed,
		PromptClass: promptClass,
		Text:        text,
		Score:       semantic.Score,
		Passed:      semantic.Passed,
		Reasons:     append([]string(nil), semantic.Reasons...),
	})

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

func buildAdmissionRouteSemanticCoverage(samples []admissionRouteSemantic, empties []admissionRouteEmpty) []admissionRouteSemanticCoverage {
	type key struct {
		index       int
		seed        string
		promptClass string
	}
	coverage := make(map[key]*admissionRouteSemanticCoverage)
	order := make([]key, 0)
	get := func(index int, seed, promptClass string) *admissionRouteSemanticCoverage {
		k := key{index: index, seed: seed, promptClass: promptClass}
		cov := coverage[k]
		if cov == nil {
			cov = &admissionRouteSemanticCoverage{Index: index, Seed: seed, PromptClass: promptClass}
			coverage[k] = cov
			order = append(order, k)
		}
		return cov
	}
	for _, s := range samples {
		promptClass := s.PromptClass
		if promptClass == "" {
			promptClass = qloopSweepPromptClass(s.Trigger, s.Seed)
		}
		cov := get(s.Index, s.Seed, promptClass)
		cov.Attempted++
		cov.Produced++
		if s.Passed {
			cov.SemanticPassed++
		} else {
			cov.SemanticMiss++
		}
		bestScore := -1
		if cov.BestScore != nil {
			bestScore = *cov.BestScore
		}
		if cov.BestRoute == "" || s.Score > bestScore {
			score := s.Score
			cov.BestRoute = s.Route
			cov.BestText = s.Text
			cov.BestScore = &score
			cov.BestPassed = s.Passed
			cov.BestReasons = append([]string(nil), s.Reasons...)
		}
	}
	for _, e := range empties {
		promptClass := qloopSweepPromptClass(e.Trigger, e.Seed)
		cov := get(e.Index, e.Seed, promptClass)
		cov.Attempted++
		cov.Empty++
	}
	sort.SliceStable(order, func(i, j int) bool {
		if order[i].index != order[j].index {
			return order[i].index < order[j].index
		}
		if order[i].seed != order[j].seed {
			return order[i].seed < order[j].seed
		}
		return order[i].promptClass < order[j].promptClass
	})
	out := make([]admissionRouteSemanticCoverage, 0, len(order))
	for _, k := range order {
		out = append(out, *coverage[k])
	}
	return out
}

func summarizeAdmissionRouteSemanticCoverage(coverage []admissionRouteSemanticCoverage) (bool, []string) {
	if len(coverage) == 0 {
		return false, []string{"semantic_coverage_missing"}
	}
	var reasons []string
	for _, cov := range coverage {
		seed := cov.Seed
		if seed == "" {
			seed = fmt.Sprintf("sample-%02d", cov.Index)
		}
		if cov.Produced == 0 {
			reasons = append(reasons, "no_route_candidate:"+seed)
			continue
		}
		if cov.SemanticPassed == 0 {
			reasons = append(reasons, "semantic_miss:"+seed)
		}
	}
	return len(reasons) == 0, reasons
}

func buildAdmissionRouteSemanticAdmission(coverage []admissionRouteSemanticCoverage) admissionRouteSemanticAdmission {
	out := admissionRouteSemanticAdmission{Passed: true}
	if len(coverage) == 0 {
		out.Passed = false
		out.Reasons = []string{"semantic_coverage_missing"}
		return out
	}
	for _, cov := range coverage {
		seed := cov.Seed
		if seed == "" {
			seed = fmt.Sprintf("sample-%02d", cov.Index)
		}
		review := admissionRouteSemanticAdmissionReview{
			Index:           cov.Index,
			Seed:            seed,
			PromptClass:     cov.PromptClass,
			CandidatesSeen:  cov.Produced,
			EmptyRoutes:     cov.Empty,
			AttemptedRoutes: cov.Attempted,
		}
		out.Reviews++
		switch {
		case cov.Produced == 0:
			review.Decision = "reject"
			review.Reason = "no_route_candidate"
			out.Rejected++
			out.NoCandidate++
			out.Reasons = append(out.Reasons, "no_route_candidate:"+seed)
		case cov.BestPassed:
			review.Decision = "admit"
			review.Reason = "semantic_pass"
			review.Route = cov.BestRoute
			review.Text = cov.BestText
			review.Score = cov.BestScore
			review.SemanticReasons = append([]string(nil), cov.BestReasons...)
			out.Admitted++
		default:
			review.Decision = "reject"
			review.Reason = "semantic_below_gate"
			review.Route = cov.BestRoute
			review.Text = cov.BestText
			review.Score = cov.BestScore
			review.SemanticReasons = append([]string(nil), cov.BestReasons...)
			out.Rejected++
			out.SemanticMiss++
			out.Reasons = append(out.Reasons, "semantic_below_gate:"+seed)
		}
		out.Decisions = append(out.Decisions, review)
	}
	out.Passed = out.Rejected == 0
	return out
}

func buildAdmissionRouteShadowBestRoute(admission admissionRouteSemanticAdmission) admissionRouteShadowBestRoute {
	out := admissionRouteShadowBestRoute{
		Schema:  "arianna.shadow_best_route.v1",
		Passed:  true,
		Reviews: admission.Reviews,
		ByRoute: make(map[string]admissionRouteShadowStats),
	}
	if len(admission.Decisions) == 0 {
		out.Passed = false
		out.Reasons = []string{"semantic_route_admission_missing"}
		out.ByRoute = nil
		return out
	}
	for _, d := range admission.Decisions {
		decision := admissionRouteShadowDecision{
			Index:           d.Index,
			Seed:            d.Seed,
			PromptClass:     d.PromptClass,
			Route:           d.Route,
			Text:            d.Text,
			Score:           d.Score,
			Reason:          d.Reason,
			SemanticReasons: append([]string(nil), d.SemanticReasons...),
		}
		if d.Decision == "admit" && d.Route != "" && d.Score != nil {
			out.Selected++
			out.SemanticScore += *d.Score
			out.RoutePlan = append(out.RoutePlan, decision)
			st := out.ByRoute[d.Route]
			st.Selected++
			st.SemanticScore += *d.Score
			st.PromptClasses = appendUniqueString(st.PromptClasses, d.PromptClass)
			out.ByRoute[d.Route] = st
			continue
		}
		out.Rejected++
		reason := d.Reason
		if reason == "" {
			reason = "not_admitted"
		}
		if d.Decision == "admit" {
			reason = "incomplete_shadow_selection"
			decision.Reason = reason
		}
		out.Reasons = append(out.Reasons, reason+":"+d.Seed)
		out.Rejects = append(out.Rejects, decision)
	}
	out.Passed = admission.Passed && out.Rejected == 0 && out.Selected == out.Reviews
	if len(out.ByRoute) == 0 {
		out.ByRoute = nil
	}
	if !out.Passed && len(out.Reasons) == 0 {
		out.Reasons = append(out.Reasons, admission.Reasons...)
	}
	return out
}

func appendUniqueString(values []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return values
	}
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func formatAdmissionRouteShadowBestRouteLine(shadow admissionRouteShadowBestRoute) string {
	routeParts := make([]string, 0, len(shadow.ByRoute))
	for route, stats := range shadow.ByRoute {
		routeParts = append(routeParts, fmt.Sprintf("%s:%d", route, stats.Selected))
	}
	sort.Strings(routeParts)
	routes := strings.Join(routeParts, ",")
	if routes == "" {
		routes = "none"
	}
	reasons := ""
	if len(shadow.Reasons) > 0 {
		reasons = " reasons=" + strings.Join(shadow.Reasons, ",")
	}
	return fmt.Sprintf("[admission-route-compare] shadow_best_route: passed=%t selected=%d/%d rejected=%d score=%d routes=%s%s",
		shadow.Passed, shadow.Selected, shadow.Reviews, shadow.Rejected, shadow.SemanticScore, routes, reasons)
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
