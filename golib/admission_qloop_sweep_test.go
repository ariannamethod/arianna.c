package main

import "testing"

func TestQloopSweepConfigs(t *testing.T) {
	cfgs := qloopSweepConfigs()
	if len(cfgs) != 12 ||
		cfgs[0].Name != "strict" ||
		cfgs[1].Name != "question_hint" ||
		cfgs[2].Name != "question_hint_qa" ||
		cfgs[3].Name != "question_source_qa" ||
		cfgs[4].Name != "question_source_qa_answer_qa" ||
		cfgs[5].Name != "question_source_user_arianna" ||
		cfgs[6].Name != "question_source_class_user_arianna" ||
		cfgs[7].Name != "question_source_class_target_user_arianna" ||
		cfgs[8].Name != qloopSweepTypedSourceConfigName ||
		cfgs[9].Name != "question_source_user_arianna_answer_qa" ||
		cfgs[10].Name != "question_hint_loose" ||
		cfgs[11].Name != "statement" {
		t.Fatalf("bad qloop sweep configs: %+v", cfgs)
	}
	if cfgs[1].Env["A2A_QLOOP_QUESTION_SOURCE_HINT"] != "1" {
		t.Fatalf("question_hint config missing env: %+v", cfgs[1].Env)
	}
	if cfgs[2].Env["A2A_QLOOP_QUESTION_SOURCE_HINT"] != "1" || cfgs[2].Env["A2A_QLOOP_ANSWER_FRAME"] != "1" {
		t.Fatalf("question_hint_qa config missing env: %+v", cfgs[2].Env)
	}
	if cfgs[3].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "qa" {
		t.Fatalf("question_source_qa config missing env: %+v", cfgs[3].Env)
	}
	if cfgs[4].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "qa" || cfgs[4].Env["A2A_QLOOP_ANSWER_FRAME"] != "1" {
		t.Fatalf("question_source_qa_answer_qa config missing env: %+v", cfgs[4].Env)
	}
	if cfgs[5].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" {
		t.Fatalf("question_source_user_arianna config missing env: %+v", cfgs[5].Env)
	}
	if cfgs[6].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" || cfgs[6].Env["A2A_QLOOP_SOURCE_CLASS"] != "prompt" {
		t.Fatalf("question_source_class_user_arianna config missing env: %+v", cfgs[6].Env)
	}
	if cfgs[7].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" || cfgs[7].Env["A2A_QLOOP_SOURCE_CLASS"] != "prompt" || cfgs[7].Env["A2A_QLOOP_TARGET_CLASS_HINT"] != "1" {
		t.Fatalf("question_source_class_target_user_arianna config missing env: %+v", cfgs[7].Env)
	}
	if cfgs[8].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" ||
		cfgs[8].Env["A2A_QLOOP_SOURCE_CLASS"] != "prompt" ||
		cfgs[8].Env["A2A_QLOOP_TYPED_SOURCE"] != "1" ||
		cfgs[8].Env["A2A_QLOOP_TYPED_SOURCE_CLASS"] != "qloop" {
		t.Fatalf("question_source_typed_user_arianna config missing env: %+v", cfgs[8].Env)
	}
	if cfgs[9].Env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" || cfgs[9].Env["A2A_QLOOP_ANSWER_FRAME"] != "1" {
		t.Fatalf("question_source_user_arianna_answer_qa config missing env: %+v", cfgs[9].Env)
	}
	if cfgs[10].Env["A2A_QLOOP_MIN"] != "0.30" || cfgs[10].Env["AM_ROUTE_COMPARE_FRAG"] != "16" {
		t.Fatalf("question_hint_loose config missing env: %+v", cfgs[10].Env)
	}
	if cfgs[11].Env["A2A_QLOOP_STATEMENT_ROUTES"] != "1" {
		t.Fatalf("statement config missing env: %+v", cfgs[11].Env)
	}
}

func TestQloopSweepTextStats(t *testing.T) {
	words, leak, debt := qloopSweepTextStats("what did the neighbour hear?")
	if words != 5 || leak || len(debt) != 0 {
		t.Fatalf("bad clean stats: words=%d leak=%v debt=%v", words, leak, debt)
	}
	_, leak, _ = qloopSweepTextStats("↳ qloop c1→c0 score 1.2: broken")
	if !leak {
		t.Fatal("route label leak not detected")
	}
}

func TestBuildQloopSweepSampleCoverage(t *testing.T) {
	cfgs := []admissionQloopSweepConfigSummary{
		{
			Name: "legacy",
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-identity", Seed: "field-origin", Produced: true, Text: "too short", Words: 2, SemanticScore: 0, QloopRoutes: 1},
				{Index: 2, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: false, EmptyReason: "no qloop candidate lines"},
			},
		},
		{
			Name: "user_arianna",
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-identity", Seed: "field-origin", Produced: true, Text: "the field answers quietly.", Words: 4, SemanticScore: 3, SemanticPassed: true, QloopRoutes: 2, QloopQSrc: 1, QloopTypedSrc: 1, QloopTargetCtx: 1},
				{Index: 2, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: true, Text: "my name is Mira.", Words: 4, SemanticScore: 0, SurfaceReasons: []string{"name_echo_artifact"}},
			},
		},
	}

	coverage := buildQloopSweepSampleCoverage(cfgs)
	if len(coverage) != 2 {
		t.Fatalf("bad coverage length: %+v", coverage)
	}
	if coverage[0].Index != 1 || coverage[0].Seed != "field-origin" || coverage[0].Produced != 2 || coverage[0].Clean != 1 || coverage[0].Short != 1 {
		t.Fatalf("bad first sample coverage: %+v", coverage[0])
	}
	if coverage[0].LeastDebtConfig != "user_arianna" || !coverage[0].LeastDebtClean || coverage[0].LeastDebtText != "the field answers quietly." {
		t.Fatalf("bad first sample best: %+v", coverage[0])
	}
	if coverage[0].BestSemanticConfig != "user_arianna" || coverage[0].BestSemanticScore != 3 || coverage[0].SemanticPassed != 1 {
		t.Fatalf("bad first sample semantic coverage: %+v", coverage[0])
	}
	if coverage[0].Configs[1].QloopTypedSrc != 1 {
		t.Fatalf("typed qloop source was not propagated into coverage: %+v", coverage[0].Configs)
	}
	if coverage[0].Configs[1].QloopTargetCtx != 1 {
		t.Fatalf("target qloop context was not propagated into coverage: %+v", coverage[0].Configs)
	}
	if coverage[1].Produced != 1 || coverage[1].Empty != 1 || coverage[1].Clean != 0 || coverage[1].SurfaceDebt != 1 {
		t.Fatalf("bad second sample coverage: %+v", coverage[1])
	}
	if coverage[1].LeastDebtConfig != "user_arianna" || coverage[1].LeastDebtClean {
		t.Fatalf("bad second sample best: %+v", coverage[1])
	}
}

func TestQloopSweepTargetHintReview(t *testing.T) {
	cfgs := []admissionQloopSweepConfigSummary{
		{
			Name: qloopSweepClassSourceConfigName,
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-identity", Seed: "field-origin", Produced: true, Text: "the field speaks now.", Words: 4, SemanticScore: 2},
				{Index: 2, Trigger: "qloop-identity", Seed: "field-boundary", Produced: true, Text: "not the outer face.", Words: 4, SemanticScore: 3, SemanticPassed: true},
				{Index: 3, Trigger: "qloop-statement", Seed: "statement-body", Produced: true, Text: "there exists a kind.", Words: 4, SemanticScore: 3, SemanticPassed: true},
				{Index: 4, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: false, EmptyReason: "no qloop candidate lines"},
			},
		},
		{
			Name: qloopSweepTargetHintConfigName,
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-identity", Seed: "field-origin", Produced: true, Text: "my own internal trace.", Words: 4, SemanticScore: 3, SemanticPassed: true, QloopTargetCtx: 1},
				{Index: 2, Trigger: "qloop-identity", Seed: "field-boundary", Produced: true, Text: "my own internal trace.", Words: 4, SemanticScore: 3, SemanticPassed: true, QloopTargetCtx: 1},
				{Index: 3, Trigger: "qloop-statement", Seed: "statement-body", Produced: false, EmptyReason: "no qloop candidate lines", QloopTargetCtx: 1},
				{Index: 4, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: true, Text: "too thin", Words: 2, SemanticScore: 0, QloopTargetCtx: 1},
			},
		},
	}

	coverage := buildQloopSweepSampleCoverage(cfgs)
	if len(coverage) != 4 {
		t.Fatalf("bad coverage length: %+v", coverage)
	}
	if r := coverage[0].TargetHintReview; r == nil || r.Decision != "target" || r.Reason != "target_semantic_score" || r.BestConfig != qloopSweepTargetHintConfigName || r.ScoreDelta != 1 {
		t.Fatalf("target should win first sample: %+v", r)
	}
	if r := coverage[1].TargetHintReview; r == nil || r.Decision != "baseline" || r.Reason != "tie_baseline" || r.BestConfig != qloopSweepClassSourceConfigName {
		t.Fatalf("tie should roll back to baseline: %+v", r)
	}
	if r := coverage[2].TargetHintReview; r == nil || r.Decision != "baseline" || r.Reason != "target_not_clean" || r.BestText != "there exists a kind." || r.TargetProduced {
		t.Fatalf("missing target should roll back to baseline: %+v", r)
	}
	if r := coverage[3].TargetHintReview; r == nil || r.Decision != "no_candidate" || r.Reason != "no_clean_candidate" || r.BestConfig != "" || !r.TargetProduced {
		t.Fatalf("dirty pair should have no candidate: %+v", r)
	}
	rollup := summarizeQloopTargetHintReviews(coverage)
	if rollup == nil || rollup.Reviews != 4 || rollup.TargetKept != 1 || rollup.RolledBack != 2 || rollup.TieRolledBack != 1 || rollup.NoCandidate != 1 || rollup.TargetMissing != 1 || rollup.BaselineMissing != 1 {
		t.Fatalf("bad target-hint rollup: %+v", rollup)
	}
}

func TestQloopSweepTypedSourceReview(t *testing.T) {
	cfgs := []admissionQloopSweepConfigSummary{
		{
			Name: qloopSweepClassSourceConfigName,
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-qloop", Seed: "same-wave", Produced: true, Text: "that's both.", Words: 2, SemanticScore: 0},
				{Index: 2, Trigger: "qloop-qloop", Seed: "same-wave-clean", Produced: true, Text: "not one wave.", Words: 3, SemanticScore: 3, SemanticPassed: true},
				{Index: 3, Trigger: "qloop-identity", Seed: "field-origin", Produced: true, Text: "not the outer face.", Words: 4, SemanticScore: 3, SemanticPassed: true},
				{Index: 4, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: false, EmptyReason: "no qloop candidate lines"},
			},
		},
		{
			Name: qloopSweepTypedSourceConfigName,
			Samples: []admissionQloopSweepSampleSummary{
				{Index: 1, Trigger: "qloop-qloop", Seed: "same-wave", Produced: true, Text: "not one wave.", Words: 3, SemanticScore: 3, SemanticPassed: true, QloopTypedSrc: 1},
				{Index: 2, Trigger: "qloop-qloop", Seed: "same-wave-clean", Produced: true, Text: "both an echo.", Words: 3, SemanticScore: 3, SemanticPassed: true, QloopTypedSrc: 1},
				{Index: 3, Trigger: "qloop-identity", Seed: "field-origin", Produced: false, EmptyReason: "no qloop candidate lines", QloopTypedSrc: 1},
				{Index: 4, Trigger: "qloop-polyphony", Seed: "many-minds", Produced: true, Text: "too thin", Words: 2, SemanticScore: 0, QloopTypedSrc: 1},
			},
		},
	}

	coverage := buildQloopSweepSampleCoverage(cfgs)
	if len(coverage) != 4 {
		t.Fatalf("bad coverage length: %+v", coverage)
	}
	if r := coverage[0].TypedSourceReview; r == nil || r.Decision != "typed_source" || r.Reason != "typed_source_only_clean" || r.BestConfig != qloopSweepTypedSourceConfigName || r.ScoreDelta != 3 {
		t.Fatalf("typed source should win dirty baseline: %+v", r)
	}
	if r := coverage[1].TypedSourceReview; r == nil || r.Decision != "baseline" || r.Reason != "tie_baseline" || r.BestConfig != qloopSweepClassSourceConfigName {
		t.Fatalf("typed tie should roll back to baseline: %+v", r)
	}
	if r := coverage[2].TypedSourceReview; r == nil || r.Decision != "baseline" || r.Reason != "typed_source_not_clean" || r.BestText != "not the outer face." || r.CandidateProduced {
		t.Fatalf("missing typed source should roll back to baseline: %+v", r)
	}
	if r := coverage[3].TypedSourceReview; r == nil || r.Decision != "no_candidate" || r.Reason != "no_clean_candidate" || r.BestConfig != "" || !r.CandidateProduced {
		t.Fatalf("dirty pair should have no candidate: %+v", r)
	}
	rollup := summarizeQloopTypedSourceReviews(coverage)
	if rollup == nil || rollup.Reviews != 4 || rollup.CandidateKept != 1 || rollup.RolledBack != 2 || rollup.TieRolledBack != 1 || rollup.NoCandidate != 1 || rollup.CandidateMissing != 1 || rollup.BaselineMissing != 1 {
		t.Fatalf("bad typed-source rollup: %+v", rollup)
	}

	bestOf := buildQloopTypedSourceBestOf(coverage, 4, 3.0)
	if bestOf == nil || !bestOf.Synthetic || bestOf.Name != "synthetic_scoped_typed_rescue" {
		t.Fatalf("bad typed-source best-of summary: %+v", bestOf)
	}
	if bestOf.Attempted != 4 || bestOf.Produced != 3 || bestOf.Empty != 1 || bestOf.SemanticPassed != 3 || bestOf.SemanticScore != 9 || bestOf.ShortCandidates != 0 || bestOf.SurfaceDebt != 0 || bestOf.QloopTypedSrc != 1 {
		t.Fatalf("bad typed-source best-of counts: %+v", bestOf)
	}
	if len(bestOf.QualityReasons) != 1 || bestOf.QualityReasons[0] != "produced_below_4" || bestOf.QualityPassed {
		t.Fatalf("best-of should stay coverage-failed: passed=%v reasons=%v", bestOf.QualityPassed, bestOf.QualityReasons)
	}
	if bestOf.Samples[0].Text != "not one wave." || bestOf.Samples[1].Text != "not one wave." || bestOf.Samples[2].Text != "not the outer face." || bestOf.Samples[3].Produced {
		t.Fatalf("bad typed-source best-of samples: %+v", bestOf.Samples)
	}

	bestOf = buildQloopTypedSourceBestOf(coverage, 3, 3.0)
	if bestOf == nil || !bestOf.QualityPassed || len(bestOf.QualityReasons) != 0 {
		t.Fatalf("best-of should pass at lower coverage threshold: %+v", bestOf)
	}
}

func TestQloopSweepSemanticAssessment(t *testing.T) {
	cases := []struct {
		name        string
		text        string
		promptClass string
		wantPass    bool
		minScore    int
		wantReason  string
	}{
		{
			name:        "cold-reader thin boundary",
			text:        "not a human.",
			promptClass: "cold-reader",
			wantPass:    false,
			minScore:    1,
			wantReason:  "nonhuman_boundary",
		},
		{
			name:        "recipient boundary",
			text:        "this person exists.",
			promptClass: "recipient-lock",
			wantPass:    true,
			minScore:    3,
			wantReason:  "recipient_boundary",
		},
		{
			name:        "conditional fragment",
			text:        "If yes the field.",
			promptClass: "recipient-lock",
			wantPass:    false,
			minScore:    0,
			wantReason:  "conditional_fragment",
		},
		{
			name:        "identity internal trace",
			text:        "my own internal trace.",
			promptClass: "identity",
			wantPass:    true,
			minScore:    3,
			wantReason:  "boundary_anchor",
		},
		{
			name:        "qloop relation",
			text:        "if they're identical wave or neither.",
			promptClass: "qloop",
			wantPass:    true,
			minScore:    3,
			wantReason:  "qloop_anchor",
		},
		{
			name:        "qloop one wave relation",
			text:        "not one wave.",
			promptClass: "qloop",
			wantPass:    true,
			minScore:    3,
			wantReason:  "question_relation",
		},
		{
			name:        "qloop echo relation",
			text:        "both an echo.",
			promptClass: "qloop",
			wantPass:    true,
			minScore:    3,
			wantReason:  "question_relation",
		},
		{
			name:        "qloop contextless both",
			text:        "that's both.",
			promptClass: "qloop",
			wantPass:    false,
			minScore:    0,
			wantReason:  "question_relation",
		},
		{
			name:        "statement constraint",
			text:        "The body remembers its own function without being.",
			promptClass: "statement",
			wantPass:    true,
			minScore:    3,
			wantReason:  "statement_anchor",
		},
		{
			name:        "polyphony truncation",
			text:        "for memory and rec.",
			promptClass: "polyphony",
			wantPass:    false,
			minScore:    0,
			wantReason:  "truncated_semantic",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := qloopSweepSemanticAssessment(tc.text, tc.promptClass)
			if got.Passed != tc.wantPass || got.Score < tc.minScore {
				t.Fatalf("bad semantic assessment: %+v", got)
			}
			if !hasString(got.Reasons, tc.wantReason) {
				t.Fatalf("missing reason %q in %+v", tc.wantReason, got)
			}
		})
	}
}

func hasString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func TestQloopSweepTextStatsSurfaceDebt(t *testing.T) {
	_, _, debt := qloopSweepTextStats("you from The My Name—. / my identity as — “.” This phrase.")
	want := map[string]bool{
		"you_from_artifact":    true,
		"name_phrase_artifact": true,
		"dangling_dash":        true,
		"slash_join":           true,
		"empty_quote":          true,
		"meta_phrase_artifact": true,
	}
	for _, reason := range debt {
		delete(want, reason)
	}
	if len(want) != 0 {
		t.Fatalf("missing debt reasons: %v in %v", want, debt)
	}

	_, _, debt = qloopSweepTextStats("you’s being.")
	if len(debt) != 1 || debt[0] != "bad_contraction" {
		t.Fatalf("bad contraction debt not isolated: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("Oleg—or—and perhaps.")
	want = map[string]bool{
		"short_dash_fragment":      true,
		"recipient_frame_artifact": true,
		"joiner_artifact":          true,
	}
	for _, reason := range debt {
		delete(want, reason)
	}
	if len(want) != 0 {
		t.Fatalf("recipient/joiner debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("—: the person inside you speaks directly.")
	if len(debt) != 1 || debt[0] != "leading_dash" {
		t.Fatalf("leading dash debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("this, yes it— you’re.")
	if len(debt) != 1 || debt[0] != "short_dash_fragment" {
		t.Fatalf("short dash fragment debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("an unknown or another.")
	if len(debt) != 1 || debt[0] != "placeholder_choice" {
		t.Fatalf("placeholder choice debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("or, to yourself.")
	if len(debt) != 1 || debt[0] != "leading_joiner_fragment" {
		t.Fatalf("leading joiner fragment debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("My Name, Mira.")
	if len(debt) != 1 || debt[0] != "name_echo_artifact" {
		t.Fatalf("name echo artifact debt not detected: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("My name is Arianna.")
	if len(debt) != 0 {
		t.Fatalf("Arianna self-name should not be name echo debt: %v", debt)
	}

	_, _, debt = qloopSweepTextStats("you have lived.")
	if len(debt) != 1 || debt[0] != "recipient_frame_artifact" {
		t.Fatalf("recipient role inversion debt not detected: %v", debt)
	}
}

func TestQloopSweepQualityRejectsSurfaceDebt(t *testing.T) {
	cfg := admissionQloopSweepConfigSummary{
		Name:           "question_hint_loose",
		Produced:       2,
		PolicyPassed:   2,
		SurfaceChecked: 2,
		SurfaceDebt:    1,
		AvgWords:       9.5,
	}
	reasons := qloopSweepQualityReasons(cfg, 1, 3.0)
	if len(reasons) != 1 || reasons[0] != "surface_debt" {
		t.Fatalf("surface debt should block quality gate: %v", reasons)
	}
}

func TestQloopSweepQualityRejectsShortCandidates(t *testing.T) {
	cfg := admissionQloopSweepConfigSummary{
		Name:            "question_hint_loose",
		Produced:        2,
		PolicyPassed:    2,
		SurfaceChecked:  2,
		ShortCandidates: 1,
		AvgWords:        4.0,
	}
	reasons := qloopSweepQualityReasons(cfg, 1, 3.0)
	if len(reasons) != 1 || reasons[0] != "short_candidate" {
		t.Fatalf("short qloop candidate should block quality gate: %v", reasons)
	}
}

func TestQloopSweepQualityRejectsLowCoverage(t *testing.T) {
	cfg := admissionQloopSweepConfigSummary{
		Name:           "statement",
		Produced:       1,
		PolicyPassed:   1,
		SurfaceChecked: 1,
		AvgWords:       4.0,
	}
	reasons := qloopSweepQualityReasons(cfg, 2, 3.0)
	if len(reasons) != 1 || reasons[0] != "produced_below_2" {
		t.Fatalf("low qloop coverage should block quality gate: %v", reasons)
	}
}

func TestChooseQloopSweepWinner(t *testing.T) {
	cfgs := []admissionQloopSweepConfigSummary{
		{Name: "strict", Produced: 0, QualityPassed: false, QualityReasons: []string{"produced_below_1"}},
		{Name: "statement", Produced: 2, PolicyPassed: 2, AvgWords: 4.5, QualityPassed: true},
	}
	winner, ok, reasons := chooseQloopSweepWinner(cfgs)
	if !ok || winner != "statement" || len(reasons) != 0 {
		t.Fatalf("bad winner: winner=%q ok=%v reasons=%v", winner, ok, reasons)
	}
}

func TestChooseQloopSweepWinnerKeepsClassSourceOnTargetTie(t *testing.T) {
	cfgs := []admissionQloopSweepConfigSummary{
		{Name: qloopSweepTargetHintConfigName, Produced: 2, PolicyPassed: 2, AvgWords: 3, SemanticPassed: 1, SemanticScore: 5, QualityPassed: true},
		{Name: qloopSweepTypedSourceConfigName, Produced: 2, PolicyPassed: 2, AvgWords: 3, SemanticPassed: 1, SemanticScore: 5, QualityPassed: true},
		{Name: qloopSweepClassSourceConfigName, Produced: 2, PolicyPassed: 2, AvgWords: 3, SemanticPassed: 1, SemanticScore: 5, QualityPassed: true},
	}
	winner, ok, reasons := chooseQloopSweepWinner(cfgs)
	if !ok || winner != qloopSweepClassSourceConfigName || len(reasons) != 0 {
		t.Fatalf("diagnostic tie should roll back to class source: winner=%q ok=%v reasons=%v", winner, ok, reasons)
	}

	cfgs[0].SemanticScore = 6
	winner, ok, reasons = chooseQloopSweepWinner(cfgs)
	if !ok || winner != qloopSweepTargetHintConfigName || len(reasons) != 0 {
		t.Fatalf("stronger target should win: winner=%q ok=%v reasons=%v", winner, ok, reasons)
	}

	cfgs[0].SemanticScore = 5
	cfgs[1].SemanticScore = 6
	winner, ok, reasons = chooseQloopSweepWinner(cfgs)
	if !ok || winner != qloopSweepTypedSourceConfigName || len(reasons) != 0 {
		t.Fatalf("stronger typed source should win: winner=%q ok=%v reasons=%v", winner, ok, reasons)
	}
}
