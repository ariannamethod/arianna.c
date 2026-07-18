package main

import "testing"

func TestQloopSweepConfigs(t *testing.T) {
	cfgs := qloopSweepConfigs()
	if len(cfgs) != 4 || cfgs[0].Name != "strict" || cfgs[1].Name != "question_hint" || cfgs[2].Name != "question_hint_loose" || cfgs[3].Name != "statement" {
		t.Fatalf("bad qloop sweep configs: %+v", cfgs)
	}
	if cfgs[1].Env["A2A_QLOOP_QUESTION_SOURCE_HINT"] != "1" {
		t.Fatalf("question_hint config missing env: %+v", cfgs[1].Env)
	}
	if cfgs[2].Env["A2A_QLOOP_MIN"] != "0.30" || cfgs[2].Env["AM_ROUTE_COMPARE_FRAG"] != "16" {
		t.Fatalf("question_hint_loose config missing env: %+v", cfgs[2].Env)
	}
	if cfgs[3].Env["A2A_QLOOP_STATEMENT_ROUTES"] != "1" {
		t.Fatalf("statement config missing env: %+v", cfgs[3].Env)
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
	if len(debt) != 2 {
		t.Fatalf("recipient/joiner debt not detected: %v", debt)
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
