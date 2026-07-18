package main

import "testing"

func TestAdmissionRoutePromptUsesQAFrame(t *testing.T) {
	if got := admissionRoutePrompt("Who are you?"); got != "Q: Who are you?\nA:" {
		t.Fatalf("bad qa prompt: %q", got)
	}
	if got := admissionRoutePrompt("Q: Who are you?\nA:"); got != "Q: Who are you?\nA:" {
		t.Fatalf("existing qa prompt changed: %q", got)
	}
}

func TestParseAdmissionDirectOutputRemovesPromptFrame(t *testing.T) {
	prompt := "Q: Who are you?\nA:"
	out := "loaded in 7 ms\nprompt: \"Q: Who are you?\\nA:\" (8 tokens)\n---\n" + prompt + " Arianna listens from the field.\n---\nprefill: 8 tok\n"
	if got := parseAdmissionDirectOutput(out, prompt); got != "Arianna listens from the field." {
		t.Fatalf("bad direct parse: %q", got)
	}
}

func TestParseChorusCellsDropsRejectedQloopGates(t *testing.T) {
	out := realFieldOutput + "\n  ↳ qloop gate c1→c2 [kv] score 0.100: rejected not a question   [entropy=4.0 I_Q^kv=-0.3]\n" +
		"timing: base_ms=2206 base_gen=19 base_retry=3 base_probe=12 base_rescue=1 base_fail=0 qloop_ms=0 qloop_gen=0 qloop_retry=0"
	cells := parseChorusCells(out)
	_, questions := chorusCounts(cells)
	if questions != 1 {
		t.Fatalf("rejected qloop gate leaked into questions: %+v", cells)
	}
}

func TestParseAdmissionRouteDiagnostics(t *testing.T) {
	out := "  ↳ qloop gate c1→c2 [kv] score 0.100: rejected not a question   [entropy=4.0]\n" +
		"  ↳ qloop gate c2→c3 [kv] score 0.200: rejected The ac.   [entropy=3.0 reason=surface]\n" +
		"  ↳ qloop gate c3→c1 [kv] score 0.300: rejected hidden drop   [entropy=2.0 reason=iq]\n" +
		"  timing: base_ms=2206 base_gen=19 base_retry=3 base_probe=12 base_rescue=1 base_fail=0 qloop_ms=0 qloop_gen=0 qloop_retry=0"
	diag := parseAdmissionRouteDiagnostics(out)
	if !diag.TimingSeen || diag.QloopGates != 3 || diag.QloopGateSurface != 1 || diag.QloopGateIQ != 1 || diag.BaseGenerated != 19 || diag.BaseRetries != 3 || diag.BaseProbe != 12 || diag.BaseRescue != 1 || diag.QloopGenerated != 0 {
		t.Fatalf("bad diagnostics: %+v", diag)
	}
	if got := routeEmptyHint("qloop", diag); got != "no qloop candidate lines (qloop_gen=0 qloop_retry=0 qloop_gates=3)" {
		t.Fatalf("bad empty hint: %q", got)
	}
}

func TestParseAdmissionRouteDiagnosticsIncludesQloopPickerStats(t *testing.T) {
	out := "timing: base_ms=2206 base_gen=19 base_retry=3 base_probe=12 base_rescue=1 base_fail=0 qloop_ms=0 qloop_gen=0 qloop_retry=0 qloop_routes=0 qloop_qsrc=0 qloop_ssrc=4 qloop_score_reject=0"
	diag := parseAdmissionRouteDiagnostics(out)
	if !diag.TimingSeen || !diag.QloopPickerSeen || diag.QloopRoutes != 0 || diag.QloopQSrc != 0 || diag.QloopSSrc != 4 || diag.QloopScoreDrop != 0 {
		t.Fatalf("bad qloop picker diagnostics: %+v", diag)
	}
	want := "no qloop candidate lines (qloop_gen=0 qloop_retry=0 qloop_gates=0) routes=0 qsrc=0 ssrc=4 score_drop=0"
	if got := routeEmptyHint("qloop", diag); got != want {
		t.Fatalf("bad qloop picker empty hint: %q", got)
	}
}

func TestQloopAdmissionTextChoosesSingleCandidate(t *testing.T) {
	cells := []chorusCell{
		{text: "the other-ness.", qloop: true},
		{text: "he has been alive.", qloop: true},
	}
	if got := qloopAdmissionText(cells); got != "he has been alive." {
		t.Fatalf("qloop admission should choose one clean candidate, got %q", got)
	}

	_, _, debt := qloopSweepTextStats(qloopAdmissionText(cells))
	if len(debt) != 0 {
		t.Fatalf("single qloop admission candidate should not inherit aggregation debt: %v", debt)
	}
}

func TestRecordAdmissionRouteCandidateSummarizesBuckets(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LOG", t.TempDir()+"/route.jsonl")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	summary := admissionRouteCompareSummary{
		Schema:  "arianna.dream_admission_route_compare_summary.v1",
		ByRoute: make(map[string]admissionRouteStats),
	}
	out := admissionRouteOutput{
		route:     "qloop",
		text:      "What remains if the field stays?",
		cells:     []chorusCell{{text: "What remains if the field stays?", qloop: true}},
		questions: 1,
		diag:      admissionRouteDiagnostics{QloopGenerated: 2, QloopRetries: 1, QloopRoutes: 2, QloopQSrc: 1, QloopSSrc: 3, QloopPickerSeen: true, TimingSeen: true},
	}
	if err := recordAdmissionRouteCandidate(iw, &summary, 3, out, "qloop-test", "seed", "frag"); err != nil {
		t.Fatal(err)
	}
	if summary.Candidates != 1 || summary.ByRoute["qloop"].Produced != 1 || summary.ByRoute["qloop"].QloopQuestions != 1 || summary.ByRoute["qloop"].QloopGenerated != 2 || summary.ByRoute["qloop"].QloopRoutes != 2 || summary.ByRoute["qloop"].QloopQSrc != 1 || summary.ByRoute["qloop"].QloopSSrc != 3 || summary.ByRoute["qloop"].QloopPickerSeen != 1 || summary.ByRoute["qloop"].TimingSeen != 1 {
		t.Fatalf("bad route summary: %+v", summary)
	}
}

func TestRecordAdmissionRouteCandidateKeepsQloopEmptyDiagnostics(t *testing.T) {
	summary := admissionRouteCompareSummary{
		Schema:  "arianna.dream_admission_route_compare_summary.v1",
		ByRoute: make(map[string]admissionRouteStats),
	}
	out := admissionRouteOutput{
		route:     "qloop",
		diag:      admissionRouteDiagnostics{QloopGates: 2, QloopGateSurface: 1, QloopGateIQ: 1, QloopGenerated: 0, TimingSeen: true},
		emptyHint: "no qloop candidate lines (qloop_gen=0 qloop_retry=0 qloop_gates=2)",
	}
	if err := recordAdmissionRouteCandidate(NewInnerWorld(), &summary, 2, out, "qloop-test", "seed", "frag"); err != nil {
		t.Fatal(err)
	}
	if summary.EmptyCandidates != 1 || summary.ByRoute["qloop"].Empty != 1 || summary.ByRoute["qloop"].QloopGates != 2 || summary.ByRoute["qloop"].QloopGateSurface != 1 || summary.ByRoute["qloop"].QloopGateIQ != 1 {
		t.Fatalf("bad empty summary: %+v", summary)
	}
	if len(summary.Empties) != 1 || summary.Empties[0].Reason != out.emptyHint {
		t.Fatalf("bad empty reason: %+v", summary.Empties)
	}
}
