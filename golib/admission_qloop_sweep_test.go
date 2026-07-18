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
	words, leak := qloopSweepTextStats("what did the neighbour hear?")
	if words != 5 || leak {
		t.Fatalf("bad clean stats: words=%d leak=%v", words, leak)
	}
	_, leak = qloopSweepTextStats("↳ qloop c1→c0 score 1.2: broken")
	if !leak {
		t.Fatal("route label leak not detected")
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
