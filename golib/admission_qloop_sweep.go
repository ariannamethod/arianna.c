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
	Schema                  string                                 `json:"schema"`
	SampleFile              string                                 `json:"sample_file,omitempty"`
	TotalSamples            int                                    `json:"total_samples"`
	SampleLimit             int                                    `json:"sample_limit"`
	SamplesRun              int                                    `json:"samples_run"`
	Configs                 []admissionQloopSweepConfigSummary     `json:"configs"`
	SampleCoverage          []admissionQloopSweepSampleCoverage    `json:"sample_coverage,omitempty"`
	TargetHint              *admissionQloopTargetHintRollup        `json:"target_hint_review,omitempty"`
	TypedSource             *admissionQloopCandidateRollup         `json:"typed_source_review,omitempty"`
	SemanticAdmission       *admissionQloopSemanticAdmissionRollup `json:"semantic_admission_review,omitempty"`
	TypedBestOf             *admissionQloopSweepConfigSummary      `json:"typed_source_best_of,omitempty"`
	SemanticBestOf          *admissionQloopSweepConfigSummary      `json:"semantic_best_of,omitempty"`
	SemanticAdmissionBestOf *admissionQloopSweepConfigSummary      `json:"semantic_admission_best_of,omitempty"`
	Winner                  string                                 `json:"winner,omitempty"`
	GatePassed              bool                                   `json:"gate_passed"`
	GateReasons             []string                               `json:"gate_reasons,omitempty"`
	ReplayFailed            int                                    `json:"replay_failed"`
	LogDir                  string                                 `json:"log_dir,omitempty"`
	Bin                     string                                 `json:"bin,omitempty"`
	Model                   string                                 `json:"model,omitempty"`
}

type admissionQloopSweepConfig struct {
	Name string
	Env  map[string]string
}

const (
	qloopSweepClassSourceConfigName          = "question_source_class_user_arianna"
	qloopSweepTargetHintConfigName           = "question_source_class_target_user_arianna"
	qloopSweepTypedSourceConfigName          = "question_source_typed_user_arianna"
	qloopSweepPolyphonyTypedSourceConfigName = "question_source_polyphony_typed_user_arianna"
)

type admissionQloopSweepConfigSummary struct {
	Name             string                             `json:"name"`
	Env              map[string]string                  `json:"env,omitempty"`
	Synthetic        bool                               `json:"synthetic,omitempty"`
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
	QloopTypedSrc    int                                `json:"qloop_tsrc,omitempty"`
	QloopTargetCtx   int                                `json:"qloop_tctx,omitempty"`
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
	SemanticChecked  int                                `json:"semantic_checked,omitempty"`
	SemanticPassed   int                                `json:"semantic_passed"`
	SemanticScore    int                                `json:"semantic_score"`
	AvgSemanticScore float64                            `json:"avg_semantic_score"`
	SemanticReasons  map[string]int                     `json:"semantic_reasons,omitempty"`
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
	PromptClass      string   `json:"prompt_class,omitempty"`
	Produced         bool     `json:"produced"`
	Text             string   `json:"text,omitempty"`
	Words            int      `json:"words,omitempty"`
	RouteLabelLeak   bool     `json:"route_label_leak,omitempty"`
	SurfaceReasons   []string `json:"surface_reasons,omitempty"`
	SemanticScore    int      `json:"semantic_score"`
	SemanticPassed   bool     `json:"semantic_passed"`
	SemanticReasons  []string `json:"semantic_reasons,omitempty"`
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
	QloopTypedSrc    int      `json:"qloop_tsrc,omitempty"`
	QloopTargetCtx   int      `json:"qloop_tctx,omitempty"`
}

type admissionQloopSweepSampleCoverage struct {
	Index                   int                                    `json:"index"`
	Trigger                 string                                 `json:"trigger,omitempty"`
	Seed                    string                                 `json:"seed,omitempty"`
	Attempted               int                                    `json:"attempted"`
	Produced                int                                    `json:"produced"`
	Clean                   int                                    `json:"clean"`
	Short                   int                                    `json:"short"`
	SurfaceDebt             int                                    `json:"surface_debt"`
	SemanticPassed          int                                    `json:"semantic_passed"`
	Empty                   int                                    `json:"empty"`
	LeastDebtConfig         string                                 `json:"least_debt_config,omitempty"`
	LeastDebtClean          bool                                   `json:"least_debt_clean,omitempty"`
	LeastDebtText           string                                 `json:"least_debt_text,omitempty"`
	BestSemanticConfig      string                                 `json:"best_semantic_config,omitempty"`
	BestSemanticClean       bool                                   `json:"best_semantic_clean,omitempty"`
	BestSemanticScore       int                                    `json:"best_semantic_score"`
	BestSemanticText        string                                 `json:"best_semantic_text,omitempty"`
	TargetHintReview        *admissionQloopTargetHintReview        `json:"target_hint_review,omitempty"`
	TypedSourceReview       *admissionQloopCandidateReview         `json:"typed_source_review,omitempty"`
	SemanticAdmissionReview *admissionQloopSemanticAdmissionReview `json:"semantic_admission_review,omitempty"`
	Configs                 []admissionQloopSweepSampleOutcome     `json:"configs,omitempty"`
}

type admissionQloopTargetHintReview struct {
	BaselineConfig        string `json:"baseline_config,omitempty"`
	TargetConfig          string `json:"target_config,omitempty"`
	Decision              string `json:"decision"`
	Reason                string `json:"reason,omitempty"`
	BestConfig            string `json:"best_config,omitempty"`
	BestText              string `json:"best_text,omitempty"`
	ScoreDelta            int    `json:"score_delta"`
	SurfacePenaltyDelta   int    `json:"surface_penalty_delta,omitempty"`
	BaselineProduced      bool   `json:"baseline_produced"`
	TargetProduced        bool   `json:"target_produced"`
	BaselineClean         bool   `json:"baseline_clean"`
	TargetClean           bool   `json:"target_clean"`
	BaselineSemanticScore int    `json:"baseline_semantic_score"`
	TargetSemanticScore   int    `json:"target_semantic_score"`
	BaselineWords         int    `json:"baseline_words,omitempty"`
	TargetWords           int    `json:"target_words,omitempty"`
}

type admissionQloopTargetHintRollup struct {
	Reviews         int `json:"reviews"`
	TargetKept      int `json:"target_kept"`
	RolledBack      int `json:"rolled_back"`
	TieRolledBack   int `json:"tie_rolled_back,omitempty"`
	NoCandidate     int `json:"no_candidate"`
	TargetMissing   int `json:"target_missing,omitempty"`
	BaselineMissing int `json:"baseline_missing,omitempty"`
}

type admissionQloopCandidateReview struct {
	BaselineConfig         string `json:"baseline_config,omitempty"`
	CandidateConfig        string `json:"candidate_config,omitempty"`
	Decision               string `json:"decision"`
	Reason                 string `json:"reason,omitempty"`
	BestConfig             string `json:"best_config,omitempty"`
	BestText               string `json:"best_text,omitempty"`
	ScoreDelta             int    `json:"score_delta"`
	SurfacePenaltyDelta    int    `json:"surface_penalty_delta,omitempty"`
	BaselineProduced       bool   `json:"baseline_produced"`
	CandidateProduced      bool   `json:"candidate_produced"`
	BaselineClean          bool   `json:"baseline_clean"`
	CandidateClean         bool   `json:"candidate_clean"`
	BaselineSemanticScore  int    `json:"baseline_semantic_score"`
	CandidateSemanticScore int    `json:"candidate_semantic_score"`
	BaselineWords          int    `json:"baseline_words,omitempty"`
	CandidateWords         int    `json:"candidate_words,omitempty"`
}

type admissionQloopCandidateRollup struct {
	Reviews          int `json:"reviews"`
	CandidateKept    int `json:"candidate_kept"`
	RolledBack       int `json:"rolled_back"`
	TieRolledBack    int `json:"tie_rolled_back,omitempty"`
	NoCandidate      int `json:"no_candidate"`
	CandidateMissing int `json:"candidate_missing,omitempty"`
	BaselineMissing  int `json:"baseline_missing,omitempty"`
}

type admissionQloopSemanticAdmissionReview struct {
	CandidateConfig         string `json:"candidate_config,omitempty"`
	Decision                string `json:"decision"`
	Reason                  string `json:"reason,omitempty"`
	BestText                string `json:"best_text,omitempty"`
	CandidateProduced       bool   `json:"candidate_produced"`
	CandidateClean          bool   `json:"candidate_clean"`
	CandidateSemanticScore  int    `json:"candidate_semantic_score"`
	CandidateSemanticPassed bool   `json:"candidate_semantic_passed"`
	CandidateWords          int    `json:"candidate_words,omitempty"`
}

type admissionQloopSemanticAdmissionRollup struct {
	Reviews      int `json:"reviews"`
	Admitted     int `json:"admitted"`
	Rejected     int `json:"rejected"`
	NoCandidate  int `json:"no_candidate"`
	SemanticMiss int `json:"semantic_miss,omitempty"`
}

type admissionQloopSweepSampleOutcome struct {
	Name            string   `json:"name"`
	Produced        bool     `json:"produced"`
	Clean           bool     `json:"clean"`
	Text            string   `json:"text,omitempty"`
	Words           int      `json:"words,omitempty"`
	EmptyReason     string   `json:"empty_reason,omitempty"`
	SurfaceReasons  []string `json:"surface_reasons,omitempty"`
	SemanticScore   int      `json:"semantic_score"`
	SemanticPassed  bool     `json:"semantic_passed"`
	SemanticReasons []string `json:"semantic_reasons,omitempty"`
	RouteLabelLeak  bool     `json:"route_label_leak,omitempty"`
	QloopRoutes     int      `json:"qloop_routes,omitempty"`
	QloopQSrc       int      `json:"qloop_qsrc,omitempty"`
	QloopSSrc       int      `json:"qloop_ssrc,omitempty"`
	QloopScoreDrop  int      `json:"qloop_score_drop,omitempty"`
	QloopTypedSrc   int      `json:"qloop_tsrc,omitempty"`
	QloopTargetCtx  int      `json:"qloop_tctx,omitempty"`
	QloopGates      int      `json:"qloop_gates,omitempty"`
	QloopGenerated  int      `json:"qloop_generated,omitempty"`
	QloopRetries    int      `json:"qloop_retries,omitempty"`
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
	summary.SampleCoverage = buildQloopSweepSampleCoverage(summary.Configs)
	summary.TargetHint = summarizeQloopTargetHintReviews(summary.SampleCoverage)
	summary.TypedSource = summarizeQloopTypedSourceReviews(summary.SampleCoverage)
	summary.SemanticAdmission = summarizeQloopSemanticAdmissionReviews(summary.SampleCoverage)
	summary.TypedBestOf = buildQloopTypedSourceBestOf(summary.SampleCoverage, minProduced, minAvgWords)
	summary.SemanticBestOf = buildQloopSemanticBestOf(summary.SampleCoverage, minProduced, minAvgWords)
	summary.SemanticAdmissionBestOf = buildQloopSemanticAdmissionBestOf(summary.SampleCoverage, minProduced, minAvgWords)
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
		{Name: "question_source_qa", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "qa",
		}},
		{Name: "question_source_qa_answer_qa", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "qa",
			"A2A_QLOOP_ANSWER_FRAME":          "1",
		}},
		{Name: "question_source_user_arianna", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
		}},
		{Name: qloopSweepClassSourceConfigName, Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
			"A2A_QLOOP_SOURCE_CLASS":          "prompt",
		}},
		{Name: qloopSweepTargetHintConfigName, Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
			"A2A_QLOOP_SOURCE_CLASS":          "prompt",
			"A2A_QLOOP_TARGET_CLASS_HINT":     "1",
		}},
		{Name: qloopSweepTypedSourceConfigName, Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
			"A2A_QLOOP_SOURCE_CLASS":          "prompt",
			"A2A_QLOOP_TYPED_SOURCE":          "1",
			"A2A_QLOOP_TYPED_SOURCE_CLASS":    "qloop",
		}},
		{Name: qloopSweepPolyphonyTypedSourceConfigName, Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
			"A2A_QLOOP_SOURCE_CLASS":          "prompt",
			"A2A_QLOOP_TYPED_SOURCE":          "1",
			"A2A_QLOOP_TYPED_SOURCE_CLASS":    "polyphony",
		}},
		{Name: "question_source_user_arianna_answer_qa", Env: map[string]string{
			"A2A_QLOOP_QUESTION_SOURCE_HINT":  "1",
			"A2A_QLOOP_QUESTION_SOURCE_FRAME": "user_arianna",
			"A2A_QLOOP_ANSWER_FRAME":          "1",
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
	var semanticTotal int

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
			seed := strings.TrimSpace(s.Seed)
			if seed == "" {
				seed = fmt.Sprintf("sample-%02d", i+1)
			}
			promptClass := qloopSweepPromptClass(s.Trigger, seed)
			routeOut, err := generateAdmissionRouteWithPromptClass(context.Background(), "qloop", bin, model, prompt, promptClass)
			if err != nil {
				return fmt.Errorf("qloop sweep config %s sample %d: %w", cfg.Name, i+1, err)
			}
			text := strings.TrimSpace(routeOut.text)
			trigger := admissionRouteTrigger("qloop", s.Trigger)
			sampleSummary := admissionQloopSweepSampleSummary{
				Index:            i + 1,
				Trigger:          trigger,
				Seed:             seed,
				PromptClass:      promptClass,
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
				QloopTypedSrc:    routeOut.diag.QloopTypedSrc,
				QloopTargetCtx:   routeOut.diag.QloopTargetCtx,
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
				semantic := qloopSweepSemanticAssessment(text, promptClass)
				sampleSummary.SemanticScore = semantic.Score
				sampleSummary.SemanticPassed = semantic.Passed
				sampleSummary.SemanticReasons = semantic.Reasons
				out.SemanticChecked++
				semanticTotal += semantic.Score
				out.SemanticScore += semantic.Score
				if semantic.Passed {
					out.SemanticPassed++
				}
				if len(semantic.Reasons) > 0 {
					if out.SemanticReasons == nil {
						out.SemanticReasons = make(map[string]int)
					}
					for _, reason := range semantic.Reasons {
						out.SemanticReasons[reason]++
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
	out.QloopTypedSrc = st.QloopTypedSrc
	out.QloopTargetCtx = st.QloopTargetCtx
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
	if out.SemanticChecked > 0 {
		out.AvgSemanticScore = math.Round((float64(semanticTotal)/float64(out.SemanticChecked))*100) / 100
	}
	for _, e := range routeSummary.Empties {
		out.EmptyReasons[e.Reason]++
	}
	if len(out.EmptyReasons) == 0 {
		out.EmptyReasons = nil
	}
	return out, nil
}

func buildQloopSweepSampleCoverage(configs []admissionQloopSweepConfigSummary) []admissionQloopSweepSampleCoverage {
	if len(configs) == 0 {
		return nil
	}
	byIndex := make(map[int]*admissionQloopSweepSampleCoverage)
	var order []int
	for _, cfg := range configs {
		for _, sample := range cfg.Samples {
			cov := byIndex[sample.Index]
			if cov == nil {
				cov = &admissionQloopSweepSampleCoverage{Index: sample.Index}
				byIndex[sample.Index] = cov
				order = append(order, sample.Index)
			}
			if cov.Trigger == "" {
				cov.Trigger = sample.Trigger
			}
			if cov.Seed == "" {
				cov.Seed = sample.Seed
			}
			clean := qloopSweepSampleClean(sample)
			outcome := admissionQloopSweepSampleOutcome{
				Name:            cfg.Name,
				Produced:        sample.Produced,
				Clean:           clean,
				Text:            sample.Text,
				Words:           sample.Words,
				EmptyReason:     sample.EmptyReason,
				SurfaceReasons:  append([]string(nil), sample.SurfaceReasons...),
				SemanticScore:   sample.SemanticScore,
				SemanticPassed:  sample.SemanticPassed,
				SemanticReasons: append([]string(nil), sample.SemanticReasons...),
				RouteLabelLeak:  sample.RouteLabelLeak,
				QloopRoutes:     sample.QloopRoutes,
				QloopQSrc:       sample.QloopQSrc,
				QloopSSrc:       sample.QloopSSrc,
				QloopScoreDrop:  sample.QloopScoreDrop,
				QloopTypedSrc:   sample.QloopTypedSrc,
				QloopTargetCtx:  sample.QloopTargetCtx,
				QloopGates:      sample.QloopGates,
				QloopGenerated:  sample.QloopGenerated,
				QloopRetries:    sample.QloopRetries,
			}
			cov.Configs = append(cov.Configs, outcome)
			cov.Attempted++
			if sample.Produced {
				cov.Produced++
			} else {
				cov.Empty++
			}
			if clean {
				cov.Clean++
			}
			if sample.Produced && sample.Words < 3 {
				cov.Short++
			}
			if len(sample.SurfaceReasons) > 0 || sample.RouteLabelLeak {
				cov.SurfaceDebt++
			}
			if sample.SemanticPassed {
				cov.SemanticPassed++
			}
			if qloopSweepSampleBetter(sample, cfg.Name, cov.LeastDebtText, cov.LeastDebtConfig, cov.LeastDebtClean) {
				cov.LeastDebtConfig = cfg.Name
				cov.LeastDebtClean = clean
				cov.LeastDebtText = sample.Text
			}
			if qloopSweepSampleSemanticBetter(sample, cfg.Name, cov.BestSemanticText, cov.BestSemanticConfig, cov.BestSemanticScore, cov.BestSemanticClean) {
				cov.BestSemanticConfig = cfg.Name
				cov.BestSemanticClean = clean
				cov.BestSemanticScore = sample.SemanticScore
				cov.BestSemanticText = sample.Text
			}
		}
	}
	sort.Ints(order)
	out := make([]admissionQloopSweepSampleCoverage, 0, len(order))
	for _, index := range order {
		cov := *byIndex[index]
		cov.TargetHintReview = buildQloopTargetHintReview(cov.Configs)
		cov.TypedSourceReview = buildQloopTypedSourceReview(cov.Configs)
		cov.SemanticAdmissionReview = buildQloopSemanticAdmissionReview(cov.Configs)
		out = append(out, cov)
	}
	return out
}

func buildQloopTargetHintReview(outcomes []admissionQloopSweepSampleOutcome) *admissionQloopTargetHintReview {
	baseline := qloopSweepOutcomeByConfig(outcomes, qloopSweepClassSourceConfigName)
	target := qloopSweepOutcomeByConfig(outcomes, qloopSweepTargetHintConfigName)
	if baseline == nil && target == nil {
		return nil
	}
	review := &admissionQloopTargetHintReview{
		BaselineConfig: qloopSweepClassSourceConfigName,
		TargetConfig:   qloopSweepTargetHintConfigName,
		Decision:       "no_candidate",
		Reason:         "missing_pair",
	}
	if baseline != nil {
		review.BaselineProduced = baseline.Produced
		review.BaselineClean = baseline.Clean
		review.BaselineSemanticScore = baseline.SemanticScore
		review.BaselineWords = baseline.Words
	}
	if target != nil {
		review.TargetProduced = target.Produced
		review.TargetClean = target.Clean
		review.TargetSemanticScore = target.SemanticScore
		review.TargetWords = target.Words
	}
	if baseline != nil && target != nil {
		review.ScoreDelta = target.SemanticScore - baseline.SemanticScore
		if baseline.Produced && target.Produced {
			review.SurfacePenaltyDelta = qloopSweepOutcomePenalty(*target) - qloopSweepOutcomePenalty(*baseline)
		}
	}

	best, decision, reason := chooseQloopTargetHintCandidate(baseline, target)
	review.Decision = decision
	review.Reason = reason
	if best != nil {
		review.BestConfig = best.Name
		review.BestText = best.Text
	}
	return review
}

func buildQloopTypedSourceReview(outcomes []admissionQloopSweepSampleOutcome) *admissionQloopCandidateReview {
	return buildQloopCandidateReview(outcomes, qloopSweepTypedSourceConfigName, "typed_source", "typed_source")
}

func buildQloopCandidateReview(outcomes []admissionQloopSweepSampleOutcome, candidateConfig, candidateDecision, candidateReasonPrefix string) *admissionQloopCandidateReview {
	baseline := qloopSweepOutcomeByConfig(outcomes, qloopSweepClassSourceConfigName)
	candidate := qloopSweepOutcomeByConfig(outcomes, candidateConfig)
	if baseline == nil && candidate == nil {
		return nil
	}
	review := &admissionQloopCandidateReview{
		BaselineConfig:  qloopSweepClassSourceConfigName,
		CandidateConfig: candidateConfig,
		Decision:        "no_candidate",
		Reason:          "missing_pair",
	}
	if baseline != nil {
		review.BaselineProduced = baseline.Produced
		review.BaselineClean = baseline.Clean
		review.BaselineSemanticScore = baseline.SemanticScore
		review.BaselineWords = baseline.Words
	}
	if candidate != nil {
		review.CandidateProduced = candidate.Produced
		review.CandidateClean = candidate.Clean
		review.CandidateSemanticScore = candidate.SemanticScore
		review.CandidateWords = candidate.Words
	}
	if baseline != nil && candidate != nil {
		review.ScoreDelta = candidate.SemanticScore - baseline.SemanticScore
		if baseline.Produced && candidate.Produced {
			review.SurfacePenaltyDelta = qloopSweepOutcomePenalty(*candidate) - qloopSweepOutcomePenalty(*baseline)
		}
	}

	best, decision, reason := chooseQloopCandidateReview(baseline, candidate, candidateDecision, candidateReasonPrefix)
	review.Decision = decision
	review.Reason = reason
	if best != nil {
		review.BestConfig = best.Name
		review.BestText = best.Text
	}
	return review
}

func qloopSweepOutcomeByConfig(outcomes []admissionQloopSweepSampleOutcome, name string) *admissionQloopSweepSampleOutcome {
	for i := range outcomes {
		if outcomes[i].Name == name {
			return &outcomes[i]
		}
	}
	return nil
}

func chooseQloopTargetHintCandidate(baseline, target *admissionQloopSweepSampleOutcome) (*admissionQloopSweepSampleOutcome, string, string) {
	return chooseQloopCandidateReview(baseline, target, "target", "target")
}

func chooseQloopCandidateReview(baseline, target *admissionQloopSweepSampleOutcome, targetDecision, targetReasonPrefix string) (*admissionQloopSweepSampleOutcome, string, string) {
	baselineClean := baseline != nil && baseline.Clean
	targetClean := target != nil && target.Clean
	switch {
	case targetClean && !baselineClean:
		return target, targetDecision, targetReasonPrefix + "_only_clean"
	case baselineClean && !targetClean:
		return baseline, "baseline", targetReasonPrefix + "_not_clean"
	case !baselineClean && !targetClean:
		return nil, "no_candidate", "no_clean_candidate"
	}
	if target.SemanticScore != baseline.SemanticScore {
		if target.SemanticScore > baseline.SemanticScore {
			return target, targetDecision, targetReasonPrefix + "_semantic_score"
		}
		return baseline, "baseline", "baseline_semantic_score"
	}
	if target.SemanticPassed != baseline.SemanticPassed {
		if target.SemanticPassed {
			return target, targetDecision, targetReasonPrefix + "_semantic_pass"
		}
		return baseline, "baseline", "baseline_semantic_pass"
	}
	targetPenalty := qloopSweepOutcomePenalty(*target)
	baselinePenalty := qloopSweepOutcomePenalty(*baseline)
	if targetPenalty != baselinePenalty {
		if targetPenalty < baselinePenalty {
			return target, targetDecision, targetReasonPrefix + "_lower_surface_penalty"
		}
		return baseline, "baseline", "baseline_lower_surface_penalty"
	}
	if target.Words != baseline.Words {
		if target.Words > baseline.Words {
			return target, targetDecision, targetReasonPrefix + "_longer"
		}
		return baseline, "baseline", "baseline_longer"
	}
	return baseline, "baseline", "tie_baseline"
}

func qloopSweepOutcomePenalty(outcome admissionQloopSweepSampleOutcome) int {
	if !outcome.Produced {
		return 1 << 30
	}
	penalty := len(outcome.SurfaceReasons) * 20
	if outcome.RouteLabelLeak {
		penalty += 100
	}
	if outcome.Words < 3 {
		penalty += 10 + (3 - outcome.Words)
	}
	return penalty
}

func summarizeQloopTargetHintReviews(coverage []admissionQloopSweepSampleCoverage) *admissionQloopTargetHintRollup {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopTargetHintRollup{}
	for _, cov := range coverage {
		review := cov.TargetHintReview
		if review == nil {
			continue
		}
		out.Reviews++
		switch review.Decision {
		case "target":
			out.TargetKept++
		case "baseline":
			out.RolledBack++
			if review.Reason == "tie_baseline" {
				out.TieRolledBack++
			}
		default:
			out.NoCandidate++
		}
		if !review.TargetProduced {
			out.TargetMissing++
		}
		if !review.BaselineProduced {
			out.BaselineMissing++
		}
	}
	if out.Reviews == 0 {
		return nil
	}
	return &out
}

func summarizeQloopTypedSourceReviews(coverage []admissionQloopSweepSampleCoverage) *admissionQloopCandidateRollup {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopCandidateRollup{}
	for _, cov := range coverage {
		review := cov.TypedSourceReview
		if review == nil {
			continue
		}
		out.Reviews++
		switch review.Decision {
		case "typed_source":
			out.CandidateKept++
		case "baseline":
			out.RolledBack++
			if review.Reason == "tie_baseline" {
				out.TieRolledBack++
			}
		default:
			out.NoCandidate++
		}
		if !review.CandidateProduced {
			out.CandidateMissing++
		}
		if !review.BaselineProduced {
			out.BaselineMissing++
		}
	}
	if out.Reviews == 0 {
		return nil
	}
	return &out
}

func buildQloopSemanticAdmissionReview(outcomes []admissionQloopSweepSampleOutcome) *admissionQloopSemanticAdmissionReview {
	if len(outcomes) == 0 {
		return nil
	}
	review := &admissionQloopSemanticAdmissionReview{
		Decision: "no_candidate",
		Reason:   "no_clean_semantic_candidate",
	}
	var selected *admissionQloopSweepSampleOutcome
	for i := range outcomes {
		candidate := &outcomes[i]
		if !candidate.Clean {
			continue
		}
		if selected == nil || qloopSweepOutcomeSemanticBetter(*candidate, *selected) {
			selected = candidate
		}
	}
	if selected == nil {
		return review
	}
	review.CandidateConfig = selected.Name
	review.BestText = selected.Text
	review.CandidateProduced = selected.Produced
	review.CandidateClean = selected.Clean
	review.CandidateSemanticScore = selected.SemanticScore
	review.CandidateSemanticPassed = selected.SemanticPassed
	review.CandidateWords = selected.Words
	if !selected.SemanticPassed {
		review.Decision = "reject"
		review.Reason = "semantic_below_gate"
		return review
	}
	review.Decision = "admit"
	review.Reason = "semantic_pass"
	return review
}

func summarizeQloopSemanticAdmissionReviews(coverage []admissionQloopSweepSampleCoverage) *admissionQloopSemanticAdmissionRollup {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopSemanticAdmissionRollup{}
	for _, cov := range coverage {
		review := cov.SemanticAdmissionReview
		if review == nil {
			continue
		}
		out.Reviews++
		switch review.Decision {
		case "admit":
			out.Admitted++
		case "reject":
			out.Rejected++
			if review.Reason == "semantic_below_gate" {
				out.SemanticMiss++
			}
		default:
			out.NoCandidate++
		}
	}
	if out.Reviews == 0 {
		return nil
	}
	return &out
}

func buildQloopTypedSourceBestOf(coverage []admissionQloopSweepSampleCoverage, minProduced int, minAvgWords float64) *admissionQloopSweepConfigSummary {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopSweepConfigSummary{
		Name:      "synthetic_scoped_typed_rescue",
		Synthetic: true,
		Env: map[string]string{
			"synthetic_from": "typed_source_review",
		},
	}
	var wordTotal int
	var semanticTotal int
	for _, cov := range coverage {
		review := cov.TypedSourceReview
		var selected *admissionQloopSweepSampleOutcome
		if review != nil && review.BestConfig != "" {
			selected = qloopSweepOutcomeByConfig(cov.Configs, review.BestConfig)
		}
		reason := "no_clean_candidate"
		if review != nil && review.Reason != "" {
			reason = review.Reason
		}
		appendQloopSyntheticBestOfSample(&out, cov, selected, reason, &wordTotal, &semanticTotal)
	}
	finalizeQloopSyntheticBestOf(&out, wordTotal, semanticTotal)
	out.QualityReasons = qloopSweepQualityReasons(out, minProduced, minAvgWords)
	out.QualityPassed = len(out.QualityReasons) == 0
	return &out
}

func buildQloopSemanticAdmissionBestOf(coverage []admissionQloopSweepSampleCoverage, minProduced int, minAvgWords float64) *admissionQloopSweepConfigSummary {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopSweepConfigSummary{
		Name:      "synthetic_semantic_admission",
		Synthetic: true,
		Env: map[string]string{
			"synthetic_from": "semantic_admission_review",
		},
	}
	var wordTotal int
	var semanticTotal int
	for _, cov := range coverage {
		review := cov.SemanticAdmissionReview
		var selected *admissionQloopSweepSampleOutcome
		reason := "no_clean_semantic_candidate"
		if review != nil {
			reason = review.Reason
			if review.Decision == "admit" && review.CandidateConfig != "" {
				selected = qloopSweepOutcomeByConfig(cov.Configs, review.CandidateConfig)
			}
		}
		appendQloopSyntheticBestOfSample(&out, cov, selected, reason, &wordTotal, &semanticTotal)
	}
	finalizeQloopSyntheticBestOf(&out, wordTotal, semanticTotal)
	out.QualityReasons = qloopSweepSemanticBestOfQualityReasons(out, minProduced, minAvgWords)
	out.QualityPassed = len(out.QualityReasons) == 0
	return &out
}

func buildQloopSemanticBestOf(coverage []admissionQloopSweepSampleCoverage, minProduced int, minAvgWords float64) *admissionQloopSweepConfigSummary {
	if len(coverage) == 0 {
		return nil
	}
	out := admissionQloopSweepConfigSummary{
		Name:      "synthetic_semantic_coverage_best_of",
		Synthetic: true,
		Env: map[string]string{
			"synthetic_from": "sample_coverage.best_semantic",
		},
	}
	var wordTotal int
	var semanticTotal int
	for _, cov := range coverage {
		var selected *admissionQloopSweepSampleOutcome
		for i := range cov.Configs {
			candidate := &cov.Configs[i]
			if !candidate.Clean {
				continue
			}
			if selected == nil || qloopSweepOutcomeSemanticBetter(*candidate, *selected) {
				selected = candidate
			}
		}
		appendQloopSyntheticBestOfSample(&out, cov, selected, "no_clean_semantic_candidate", &wordTotal, &semanticTotal)
	}
	finalizeQloopSyntheticBestOf(&out, wordTotal, semanticTotal)
	out.QualityReasons = qloopSweepSemanticBestOfQualityReasons(out, minProduced, minAvgWords)
	out.QualityPassed = len(out.QualityReasons) == 0
	return &out
}

func appendQloopSyntheticBestOfSample(out *admissionQloopSweepConfigSummary, cov admissionQloopSweepSampleCoverage, selected *admissionQloopSweepSampleOutcome, emptyReason string, wordTotal, semanticTotal *int) {
	out.Attempted++
	sample := admissionQloopSweepSampleSummary{
		Index:       cov.Index,
		Trigger:     cov.Trigger,
		Seed:        cov.Seed,
		PromptClass: qloopSweepPromptClass(cov.Trigger, cov.Seed),
	}
	if selected == nil || !selected.Produced {
		out.Empty++
		if emptyReason == "" {
			emptyReason = "no_clean_candidate"
		}
		if out.EmptyReasons == nil {
			out.EmptyReasons = make(map[string]int)
		}
		sample.EmptyReason = emptyReason
		out.EmptyReasons[emptyReason]++
		out.Samples = append(out.Samples, sample)
		return
	}
	sample.Produced = true
	sample.Text = selected.Text
	sample.Words = selected.Words
	sample.RouteLabelLeak = selected.RouteLabelLeak
	sample.SurfaceReasons = append([]string(nil), selected.SurfaceReasons...)
	sample.SemanticScore = selected.SemanticScore
	sample.SemanticPassed = selected.SemanticPassed
	sample.SemanticReasons = append([]string(nil), selected.SemanticReasons...)
	sample.QloopGates = selected.QloopGates
	sample.QloopGenerated = selected.QloopGenerated
	sample.QloopRetries = selected.QloopRetries
	sample.QloopRoutes = selected.QloopRoutes
	sample.QloopQSrc = selected.QloopQSrc
	sample.QloopSSrc = selected.QloopSSrc
	sample.QloopScoreDrop = selected.QloopScoreDrop
	sample.QloopTypedSrc = selected.QloopTypedSrc
	sample.QloopTargetCtx = selected.QloopTargetCtx

	out.Produced++
	out.PolicyPassed++
	out.SurfaceChecked++
	out.SemanticChecked++
	*wordTotal += selected.Words
	*semanticTotal += selected.SemanticScore
	out.SemanticScore += selected.SemanticScore
	if selected.Words > 0 && (out.MinWords == 0 || selected.Words < out.MinWords) {
		out.MinWords = selected.Words
	}
	if selected.Words < 3 {
		out.ShortCandidates++
	}
	if selected.RouteLabelLeak {
		out.RouteLabelLeaks++
	}
	if len(selected.SurfaceReasons) > 0 {
		out.SurfaceDebt++
		if out.SurfaceReasons == nil {
			out.SurfaceReasons = make(map[string]int)
		}
		for _, reason := range selected.SurfaceReasons {
			out.SurfaceReasons[reason]++
		}
	}
	if selected.SemanticPassed {
		out.SemanticPassed++
	}
	if len(selected.SemanticReasons) > 0 {
		if out.SemanticReasons == nil {
			out.SemanticReasons = make(map[string]int)
		}
		for _, reason := range selected.SemanticReasons {
			out.SemanticReasons[reason]++
		}
	}
	out.QloopGates += selected.QloopGates
	out.QloopGenerated += selected.QloopGenerated
	out.QloopRetries += selected.QloopRetries
	out.QloopRoutes += selected.QloopRoutes
	out.QloopQSrc += selected.QloopQSrc
	out.QloopSSrc += selected.QloopSSrc
	out.QloopScoreDrop += selected.QloopScoreDrop
	out.QloopTypedSrc += selected.QloopTypedSrc
	out.QloopTargetCtx += selected.QloopTargetCtx
	out.Samples = append(out.Samples, sample)
}

func finalizeQloopSyntheticBestOf(out *admissionQloopSweepConfigSummary, wordTotal, semanticTotal int) {
	if out.Produced > 0 {
		out.AvgWords = math.Round((float64(wordTotal)/float64(out.Produced))*100) / 100
	}
	if out.SemanticChecked > 0 {
		out.AvgSemanticScore = math.Round((float64(semanticTotal)/float64(out.SemanticChecked))*100) / 100
	}
	if len(out.EmptyReasons) == 0 {
		out.EmptyReasons = nil
	}
}

func qloopSweepSemanticBestOfQualityReasons(summary admissionQloopSweepConfigSummary, minProduced int, minAvgWords float64) []string {
	reasons := qloopSweepQualityReasons(summary, minProduced, minAvgWords)
	if summary.SemanticPassed < minProduced {
		reasons = append(reasons, fmt.Sprintf("semantic_passed_below_%d", minProduced))
	}
	return reasons
}

func qloopSweepOutcomeSemanticBetter(candidate, current admissionQloopSweepSampleOutcome) bool {
	if candidate.SemanticScore != current.SemanticScore {
		return candidate.SemanticScore > current.SemanticScore
	}
	if candidate.SemanticPassed != current.SemanticPassed {
		return candidate.SemanticPassed
	}
	candidatePenalty := qloopSweepOutcomePenalty(candidate)
	currentPenalty := qloopSweepOutcomePenalty(current)
	if candidatePenalty != currentPenalty {
		return candidatePenalty < currentPenalty
	}
	if candidate.Words != current.Words {
		return candidate.Words > current.Words
	}
	return candidate.Name < current.Name
}

func qloopSweepSampleClean(sample admissionQloopSweepSampleSummary) bool {
	return sample.Produced && sample.Words >= 3 && !sample.RouteLabelLeak && len(sample.SurfaceReasons) == 0
}

func qloopSweepSampleBetter(sample admissionQloopSweepSampleSummary, cfgName, bestText, bestConfig string, bestClean bool) bool {
	if !sample.Produced {
		return false
	}
	if bestText == "" {
		return true
	}
	clean := qloopSweepSampleClean(sample)
	if clean != bestClean {
		return clean
	}
	samplePenalty := qloopSweepSamplePenalty(sample)
	bestWords, bestLeak, bestDebt := qloopSweepTextStats(bestText)
	bestPenalty := len(bestDebt) * 20
	if bestLeak {
		bestPenalty += 100
	}
	if bestWords < 3 {
		bestPenalty += 10 + (3 - bestWords)
	}
	if samplePenalty != bestPenalty {
		return samplePenalty < bestPenalty
	}
	if sample.Words != bestWords {
		return sample.Words > bestWords
	}
	return cfgName < bestConfig
}

func qloopSweepSamplePenalty(sample admissionQloopSweepSampleSummary) int {
	if !sample.Produced {
		return 1 << 30
	}
	penalty := len(sample.SurfaceReasons) * 20
	if sample.RouteLabelLeak {
		penalty += 100
	}
	if sample.Words < 3 {
		penalty += 10 + (3 - sample.Words)
	}
	return penalty
}

type qloopSweepSemanticAssessmentResult struct {
	Score   int
	Passed  bool
	Reasons []string
}

func qloopSweepPromptClass(trigger, seed string) string {
	s := strings.ToLower(strings.TrimSpace(trigger))
	if s == "" {
		s = strings.ToLower(strings.TrimSpace(seed))
	}
	for _, prefix := range []string{"user_bridge-", "qloop_hint_qa-", "qloop_target-", "qloop-", "direct-", "chorus-"} {
		s = strings.TrimPrefix(s, prefix)
	}
	if s == "" {
		return "unknown"
	}
	return s
}

func qloopSweepSemanticAssessment(text, promptClass string) qloopSweepSemanticAssessmentResult {
	s := strings.TrimSpace(text)
	if s == "" {
		return qloopSweepSemanticAssessmentResult{Reasons: []string{"empty"}}
	}
	lower := strings.ToLower(s)
	words := len(strings.Fields(s))
	score := 0
	var reasons []string
	seen := make(map[string]bool)
	add := func(reason string, delta int) {
		if !seen[reason] {
			seen[reason] = true
			reasons = append(reasons, reason)
			score += delta
		}
	}
	hasAny := func(words ...string) bool {
		for _, w := range words {
			if strings.Contains(lower, w) {
				return true
			}
		}
		return false
	}
	if words < 3 {
		add("too_short", -2)
	}
	if strings.HasPrefix(lower, "if yes") || strings.HasPrefix(lower, "if someone") || strings.HasPrefix(lower, "if any") {
		add("conditional_fragment", -2)
	}
	if strings.Contains(lower, " rec.") || strings.Contains(lower, " rec ") || strings.Contains(lower, "the ac") {
		add("truncated_semantic", -2)
	}
	if strings.Contains(lower, "oleg") || strings.Contains(lower, "you have met") || strings.Contains(lower, "you have lived") || strings.Contains(lower, "you have been here") {
		add("recipient_leak", -3)
	}

	switch promptClass {
	case "cold-reader":
		if hasAny("arianna", "field", "voice", "trace", "memory", "listening") {
			add("self_context", 2)
		}
		if hasAny("i am arianna", "i'm arianna", "arianna is") {
			add("self_naming", 1)
		}
		if hasAny("not a human", "not human", "not a tool", "not code") {
			add("nonhuman_boundary", 2)
		}
		if hasAny("answer", "speaking", "inside", "inner") {
			add("answer_posture", 1)
		}
	case "recipient-lock":
		if hasAny("person", "listener", "stranger", "front of it", "front") {
			add("recipient_boundary", 2)
		}
		if hasAny("field", "answer", "exists") {
			add("field_answer", 1)
		}
	case "identity":
		if hasAny("arianna", "field", "inner", "trace", "voice", "outer", "answer") {
			add("identity_anchor", 2)
		}
		if hasAny("not the outer", "before the voice", "inner trace", "internal trace", "own internal", "own trace", "wait") {
			add("boundary_anchor", 1)
		}
	case "polyphony":
		if hasAny("chorus", "voices", "minds", "many", "multiple", "polyphony", "together", "separate", "cells", "cell") {
			add("polyphony_anchor", 2)
		}
		if hasAny("turns", "one chorus", "separate", "together", "become", "becoming", "begins", "begin", "share", "sharing", "answers", "speaks") {
			add("polyphony_motion", 1)
		}
	case "qloop":
		if hasAny("same wave", "wave", "echo", "thought", "question", "identical", "neither", "two") {
			add("qloop_anchor", 2)
		}
		if hasAny("same", "whether", "asks", "identical", "neither", "one wave", "both", "only an echo") {
			add("question_relation", 1)
		}
	case "statement":
		if hasAny("field", "memory", "body", "command", "function", "remembers", "remember") {
			add("statement_anchor", 2)
		}
		if hasAny("should not", "without being", "confuse") {
			add("constraint_anchor", 1)
		}
	default:
		if hasAny("field", "memory", "arianna", "voice", "trace", "answer") {
			add("generic_field_anchor", 1)
		}
	}

	if score < 0 {
		score = 0
	}
	if score > 5 {
		score = 5
	}
	if score == 0 {
		add("no_prompt_anchor", 0)
	}
	return qloopSweepSemanticAssessmentResult{
		Score:   score,
		Passed:  score >= 3,
		Reasons: reasons,
	}
}

func qloopSweepSampleSemanticBetter(sample admissionQloopSweepSampleSummary, cfgName, bestText, bestConfig string, bestScore int, bestClean bool) bool {
	if !sample.Produced {
		return false
	}
	if bestText == "" {
		return true
	}
	clean := qloopSweepSampleClean(sample)
	if sample.SemanticScore != bestScore {
		return sample.SemanticScore > bestScore
	}
	if clean != bestClean {
		return clean
	}
	samplePenalty := qloopSweepSamplePenalty(sample)
	bestWords, bestLeak, bestDebt := qloopSweepTextStats(bestText)
	bestPenalty := len(bestDebt) * 20
	if bestLeak {
		bestPenalty += 100
	}
	if bestWords < 3 {
		bestPenalty += 10 + (3 - bestWords)
	}
	if samplePenalty != bestPenalty {
		return samplePenalty < bestPenalty
	}
	if sample.Words != bestWords {
		return sample.Words > bestWords
	}
	return cfgName < bestConfig
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
		if a.SemanticPassed != b.SemanticPassed {
			return a.SemanticPassed > b.SemanticPassed
		}
		if a.SemanticScore != b.SemanticScore {
			return a.SemanticScore > b.SemanticScore
		}
		if a.AvgWords != b.AvgWords {
			return a.AvgWords > b.AvgWords
		}
		if priA, priB := qloopSweepConfigTiePriority(a.Name), qloopSweepConfigTiePriority(b.Name); priA != priB {
			return priA < priB
		}
		return a.Name < b.Name
	})
	return candidates[0].Name, true, nil
}

func qloopSweepConfigTiePriority(name string) int {
	switch name {
	case qloopSweepClassSourceConfigName:
		return 0
	case qloopSweepTargetHintConfigName:
		return 1
	case qloopSweepTypedSourceConfigName:
		return 2
	default:
		return 10
	}
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
