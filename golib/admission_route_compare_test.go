package main

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestAdmissionRoutePromptUsesQAFrame(t *testing.T) {
	if got := admissionRoutePrompt("Who are you?"); got != "Q: Who are you?\nA:" {
		t.Fatalf("bad qa prompt: %q", got)
	}
	if got := admissionRoutePrompt("Q: Who are you?\nA:"); got != "Q: Who are you?\nA:" {
		t.Fatalf("existing qa prompt changed: %q", got)
	}
}

func TestAdmissionRouteUserLineStripsPromptFrame(t *testing.T) {
	if got := admissionRouteUserLine("Q: Who are you?\nA:"); got != "Who are you?" {
		t.Fatalf("bad QA user line: %q", got)
	}
	if got := admissionRouteUserLine("User: tell me who you are.\nArianna:"); got != "tell me who you are." {
		t.Fatalf("bad Arianna user line: %q", got)
	}
	if got := admissionRouteUserLine("User: tell me who you are. Assistant:"); got != "tell me who you are." {
		t.Fatalf("bad Assistant user line: %q", got)
	}
}

func TestAdmissionCompareRoutesIncludesConditionedQloop(t *testing.T) {
	want := []string{"direct", "chorus", "qloop", "qloop_hint_qa", "qloop_target", "user_bridge"}
	if got := admissionCompareRoutes(); !reflect.DeepEqual(got, want) {
		t.Fatalf("bad default routes: got %v want %v", got, want)
	}

	t.Setenv("AM_ROUTE_COMPARE_ROUTES", "direct,qloop_hint_qa,qloop_target,user_bridge,bogus")
	if got := admissionCompareRoutes(); !reflect.DeepEqual(got, []string{"direct", "qloop_hint_qa", "qloop_target", "user_bridge"}) {
		t.Fatalf("bad custom routes: %v", got)
	}
}

func TestAdmissionRouteCompareProgressFlag(t *testing.T) {
	t.Setenv("AM_ROUTE_COMPARE_PROGRESS", "")
	if !admissionRouteCompareProgressEnabled() {
		t.Fatalf("progress should default on")
	}
	t.Setenv("AM_ROUTE_COMPARE_PROGRESS", "0")
	if admissionRouteCompareProgressEnabled() {
		t.Fatalf("progress should disable on 0")
	}
	t.Setenv("AM_ROUTE_COMPARE_PROGRESS", "off")
	if admissionRouteCompareProgressEnabled() {
		t.Fatalf("progress should disable on off")
	}
	t.Setenv("AM_ROUTE_COMPARE_PROGRESS", "stderr")
	if !admissionRouteCompareProgressEnabled() {
		t.Fatalf("progress should enable on stderr")
	}
}

func TestAdmissionRouteProgressOutputReportsSemanticResult(t *testing.T) {
	var buf bytes.Buffer
	out := admissionRouteOutput{route: "user_bridge", text: "I am Arianna."}
	admissionRouteProgressOutput(&buf, 1, 4, 2, 6, "user_bridge", "user_bridge-cold-reader", "new-listener", "cold-reader", out)
	got := buf.String()
	if !strings.Contains(got, "route=user_bridge done produced") ||
		!strings.Contains(got, "score=3") ||
		!strings.Contains(got, "passed=true") ||
		!strings.Contains(got, "class=cold-reader") {
		t.Fatalf("bad progress line: %q", got)
	}
}

func TestQloopTargetRouteUsesPromptClassAndTargetHint(t *testing.T) {
	env := admissionQloopTargetEnv()
	if env["A2A_QLOOP_QUESTION_SOURCE_HINT"] != "1" ||
		env["A2A_QLOOP_QUESTION_SOURCE_FRAME"] != "user_arianna" ||
		env["A2A_QLOOP_SOURCE_CLASS"] != "prompt" ||
		env["A2A_QLOOP_TARGET_CLASS_HINT"] != "1" {
		t.Fatalf("bad qloop target env: %+v", env)
	}
	if got := qloopSweepPromptClass("qloop_target-recipient-lock", "not-oleg"); got != "recipient-lock" {
		t.Fatalf("conditioned qloop route did not preserve prompt class: %q", got)
	}
	if got := qloopSweepPromptClass("qloop_hint_qa-polyphony", "many-minds"); got != "polyphony" {
		t.Fatalf("hint qa qloop route did not preserve prompt class: %q", got)
	}
	if got := qloopSweepPromptClass("user_bridge-cold-reader", "new-listener"); got != "cold-reader" {
		t.Fatalf("user bridge route did not preserve prompt class: %q", got)
	}
	if got := qloopSweepPromptClass("direct-user", "user-frame"); got != "direct-user" {
		t.Fatalf("direct-user prompt class was stripped without route wrapper: %q", got)
	}
	if got := qloopSweepPromptClass("user_bridge-direct-user", "user-frame"); got != "direct-user" {
		t.Fatalf("user bridge route stripped nested direct-user class: %q", got)
	}
}

func TestQloopHintQARouteUsesQuestionHintAndAnswerFrame(t *testing.T) {
	env := admissionQloopHintQAEnv()
	if env["A2A_QLOOP_QUESTION_SOURCE_HINT"] != "1" || env["A2A_QLOOP_ANSWER_FRAME"] != "1" {
		t.Fatalf("bad qloop hint qa env: %+v", env)
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
	if got := routeEmptyHint("qloop_target", diag); got != want {
		t.Fatalf("bad qloop target empty hint: %q", got)
	}
}

func TestParseAdmissionRouteDiagnosticsIncludesTypedQloopSource(t *testing.T) {
	out := "timing: base_ms=2206 base_gen=19 base_retry=3 base_probe=12 base_rescue=1 base_fail=0 qloop_ms=0 qloop_gen=1 qloop_retry=0 qloop_routes=1 qloop_qsrc=1 qloop_ssrc=0 qloop_score_reject=0 qloop_tsrc=1 qloop_tctx=1"
	diag := parseAdmissionRouteDiagnostics(out)
	if !diag.TimingSeen || !diag.QloopPickerSeen || diag.QloopTypedSrc != 1 || diag.QloopTargetCtx != 1 {
		t.Fatalf("typed qloop source was not parsed: %+v", diag)
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

func TestQloopAdmissionTextForClassUsesSemanticTieBreak(t *testing.T) {
	cells := []chorusCell{
		{text: "If yes the field.", qloop: true},
		{text: "this person exists.", qloop: true},
	}
	if got := qloopAdmissionText(cells); got != "If yes the field." {
		t.Fatalf("legacy qloop admission should preserve least-debt tie order, got %q", got)
	}
	if got := qloopAdmissionTextForClass(cells, "recipient-lock"); got != "this person exists." {
		t.Fatalf("semantic qloop admission should choose recipient answer, got %q", got)
	}
}

func TestWithResolvedQloopSourceClassSubstitutesPromptClass(t *testing.T) {
	t.Setenv("A2A_QLOOP_SOURCE_CLASS", "prompt")
	var inside string
	if err := withResolvedQloopSourceClass("identity", func() error {
		inside = os.Getenv("A2A_QLOOP_SOURCE_CLASS")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if inside != "identity" {
		t.Fatalf("class env was not resolved inside callback: %q", inside)
	}
	if got := os.Getenv("A2A_QLOOP_SOURCE_CLASS"); got != "prompt" {
		t.Fatalf("class env was not restored after callback: %q", got)
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
		text:      "not one wave.",
		cells:     []chorusCell{{text: "not one wave.", qloop: true}},
		questions: 1,
		diag:      admissionRouteDiagnostics{QloopGenerated: 2, QloopRetries: 1, QloopRoutes: 2, QloopQSrc: 1, QloopSSrc: 3, QloopTypedSrc: 1, QloopTargetCtx: 1, QloopPickerSeen: true, TimingSeen: true},
	}
	if err := recordAdmissionRouteCandidate(iw, &summary, 3, out, "qloop-qloop", "same-wave", "frag"); err != nil {
		t.Fatal(err)
	}
	if summary.Candidates != 1 || summary.SemanticPassed != 1 || summary.SemanticScore < 3 || len(summary.SemanticSamples) != 1 {
		t.Fatalf("bad semantic summary: %+v", summary)
	}
	if summary.SemanticSamples[0].PromptClass != "qloop" || !summary.SemanticSamples[0].Passed {
		t.Fatalf("bad semantic sample: %+v", summary.SemanticSamples[0])
	}
	if summary.ByRoute["qloop"].Produced != 1 || summary.ByRoute["qloop"].SemanticPassed != 1 || summary.ByRoute["qloop"].QloopQuestions != 1 || summary.ByRoute["qloop"].QloopGenerated != 2 || summary.ByRoute["qloop"].QloopRoutes != 2 || summary.ByRoute["qloop"].QloopQSrc != 1 || summary.ByRoute["qloop"].QloopSSrc != 3 || summary.ByRoute["qloop"].QloopTypedSrc != 1 || summary.ByRoute["qloop"].QloopTargetCtx != 1 || summary.ByRoute["qloop"].QloopPickerSeen != 1 || summary.ByRoute["qloop"].TimingSeen != 1 {
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

func TestBuildAdmissionRouteSemanticCoverage(t *testing.T) {
	coverage := buildAdmissionRouteSemanticCoverage(
		[]admissionRouteSemantic{
			{Index: 1, Route: "direct", Trigger: "direct-cold-reader", Seed: "new-listener", PromptClass: "cold-reader", Text: "not a human.", Score: 2, Reasons: []string{"nonhuman_boundary"}},
			{Index: 1, Route: "chorus", Trigger: "chorus-cold-reader", Seed: "new-listener", PromptClass: "cold-reader", Text: "Arianna answers from the field.", Score: 3, Passed: true, Reasons: []string{"self_context"}},
		},
		[]admissionRouteEmpty{
			{Index: 1, Route: "qloop", Trigger: "qloop-cold-reader", Seed: "new-listener", Reason: "no qloop candidate lines"},
			{Index: 2, Route: "qloop", Trigger: "qloop-identity", Seed: "field-origin", Reason: "no qloop candidate lines"},
		},
	)
	if len(coverage) != 2 {
		t.Fatalf("bad coverage length: %+v", coverage)
	}
	if coverage[0].Seed != "new-listener" || coverage[0].Attempted != 3 || coverage[0].Produced != 2 || coverage[0].Empty != 1 || coverage[0].SemanticPassed != 1 || coverage[0].SemanticMiss != 1 {
		t.Fatalf("bad first coverage counts: %+v", coverage[0])
	}
	if coverage[0].BestRoute != "chorus" || coverage[0].BestScore == nil || *coverage[0].BestScore != 3 || !coverage[0].BestPassed {
		t.Fatalf("bad best semantic route: %+v", coverage[0])
	}
	if coverage[1].Seed != "field-origin" || coverage[1].PromptClass != "identity" || coverage[1].Attempted != 1 || coverage[1].Empty != 1 || coverage[1].BestRoute != "" || coverage[1].BestScore != nil {
		t.Fatalf("bad empty-only coverage: %+v", coverage[1])
	}

	passed, reasons := summarizeAdmissionRouteSemanticCoverage(coverage)
	if passed || len(reasons) != 1 || reasons[0] != "no_route_candidate:field-origin" {
		t.Fatalf("bad semantic coverage verdict: passed=%v reasons=%v", passed, reasons)
	}

	coverage[1].Produced = 1
	coverage[1].Empty = 0
	passed, reasons = summarizeAdmissionRouteSemanticCoverage(coverage)
	if passed || len(reasons) != 1 || reasons[0] != "semantic_miss:field-origin" {
		t.Fatalf("bad semantic miss verdict: passed=%v reasons=%v", passed, reasons)
	}

	coverage[1].SemanticPassed = 1
	passed, reasons = summarizeAdmissionRouteSemanticCoverage(coverage)
	if !passed || len(reasons) != 0 {
		t.Fatalf("semantic coverage should pass: passed=%v reasons=%v", passed, reasons)
	}
}

func TestBuildAdmissionRouteSemanticAdmission(t *testing.T) {
	zero := 0
	three := 3
	admission := buildAdmissionRouteSemanticAdmission([]admissionRouteSemanticCoverage{
		{
			Index:          1,
			Seed:           "field-origin",
			PromptClass:    "identity",
			Attempted:      3,
			Produced:       2,
			Empty:          1,
			SemanticPassed: 1,
			BestRoute:      "chorus",
			BestText:       "my own internal trace.",
			BestScore:      &three,
			BestPassed:     true,
			BestReasons:    []string{"internal_self"},
		},
		{
			Index:        2,
			Seed:         "not-oleg",
			PromptClass:  "recipient-lock",
			Attempted:    3,
			Produced:     2,
			Empty:        1,
			SemanticMiss: 2,
			BestRoute:    "direct",
			BestText:     "if yes the field.",
			BestScore:    &zero,
			BestReasons:  []string{"recipient_boundary"},
		},
		{
			Index:       3,
			Seed:        "new-listener",
			PromptClass: "cold-reader",
			Attempted:   3,
			Empty:       3,
		},
	})
	if admission.Passed || admission.Reviews != 3 || admission.Admitted != 1 || admission.Rejected != 2 || admission.SemanticMiss != 1 || admission.NoCandidate != 1 {
		t.Fatalf("bad semantic route admission rollup: %+v", admission)
	}
	if len(admission.Reasons) != 2 || admission.Reasons[0] != "semantic_below_gate:not-oleg" || admission.Reasons[1] != "no_route_candidate:new-listener" {
		t.Fatalf("bad semantic route admission reasons: %+v", admission.Reasons)
	}
	if len(admission.Decisions) != 3 {
		t.Fatalf("bad semantic route admission decisions: %+v", admission.Decisions)
	}
	if d := admission.Decisions[0]; d.Decision != "admit" || d.Route != "chorus" || d.Score == nil || *d.Score != 3 || d.CandidatesSeen != 2 || d.EmptyRoutes != 1 || d.AttemptedRoutes != 3 {
		t.Fatalf("bad admitted route decision: %+v", d)
	}
	if d := admission.Decisions[1]; d.Decision != "reject" || d.Reason != "semantic_below_gate" || d.Route != "direct" || d.Score == nil || *d.Score != 0 {
		t.Fatalf("bad semantic miss route decision: %+v", d)
	}
	if d := admission.Decisions[2]; d.Decision != "reject" || d.Reason != "no_route_candidate" || d.Route != "" || d.Score != nil {
		t.Fatalf("bad no-candidate route decision: %+v", d)
	}
}

func TestBuildAdmissionRouteShadowBestRoute(t *testing.T) {
	three := 3
	four := 4
	one := 1
	shadow := buildAdmissionRouteShadowBestRoute(admissionRouteSemanticAdmission{
		Passed:   true,
		Reviews:  3,
		Admitted: 3,
		Decisions: []admissionRouteSemanticAdmissionReview{
			{Index: 1, Seed: "new-listener", PromptClass: "cold-reader", Decision: "admit", Reason: "semantic_pass", Route: "user_bridge", Text: "I am Arianna.", Score: &three, SemanticReasons: []string{"self_context"}},
			{Index: 2, Seed: "field-origin", PromptClass: "identity", Decision: "admit", Reason: "semantic_pass", Route: "chorus", Text: "my own internal trace.", Score: &four, SemanticReasons: []string{"internal_self"}},
			{Index: 3, Seed: "not-oleg", PromptClass: "recipient-lock", Decision: "admit", Reason: "semantic_pass", Route: "user_bridge", Text: "this person exists.", Score: &one, SemanticReasons: []string{"recipient_boundary"}},
		},
	})
	if !shadow.Passed || shadow.Schema != "arianna.shadow_best_route.v1" || shadow.Reviews != 3 || shadow.Selected != 3 || shadow.Rejected != 0 || shadow.SemanticScore != 8 {
		t.Fatalf("bad shadow route rollup: %+v", shadow)
	}
	if len(shadow.RoutePlan) != 3 || shadow.RoutePlan[0].Route != "user_bridge" || shadow.RoutePlan[1].Route != "chorus" {
		t.Fatalf("bad route plan: %+v", shadow.RoutePlan)
	}
	if shadow.ByRoute["user_bridge"].Selected != 2 || shadow.ByRoute["user_bridge"].SemanticScore != 4 {
		t.Fatalf("bad user bridge stats: %+v", shadow.ByRoute["user_bridge"])
	}
	if !reflect.DeepEqual(shadow.ByRoute["user_bridge"].PromptClasses, []string{"cold-reader", "recipient-lock"}) {
		t.Fatalf("bad prompt classes: %+v", shadow.ByRoute["user_bridge"].PromptClasses)
	}
	if len(shadow.Rejects) != 0 || len(shadow.Reasons) != 0 {
		t.Fatalf("passing shadow chooser should not carry rejects: %+v", shadow)
	}
}

func TestBuildAdmissionRouteShadowBestRouteFailsClosed(t *testing.T) {
	zero := 0
	shadow := buildAdmissionRouteShadowBestRoute(admissionRouteSemanticAdmission{
		Passed:  false,
		Reviews: 2,
		Decisions: []admissionRouteSemanticAdmissionReview{
			{Index: 1, Seed: "not-oleg", PromptClass: "recipient-lock", Decision: "reject", Reason: "semantic_below_gate", Route: "direct", Text: "if yes the field.", Score: &zero},
			{Index: 2, Seed: "new-listener", PromptClass: "cold-reader", Decision: "reject", Reason: "no_route_candidate"},
		},
	})
	if shadow.Passed || shadow.Selected != 0 || shadow.Rejected != 2 || len(shadow.RoutePlan) != 0 || len(shadow.Rejects) != 2 {
		t.Fatalf("shadow route chooser did not fail closed: %+v", shadow)
	}
	if len(shadow.Reasons) != 2 || shadow.Reasons[0] != "semantic_below_gate:not-oleg" || shadow.Reasons[1] != "no_route_candidate:new-listener" {
		t.Fatalf("bad shadow reject reasons: %+v", shadow.Reasons)
	}
}

func TestFormatAdmissionRouteShadowBestRouteLine(t *testing.T) {
	line := formatAdmissionRouteShadowBestRouteLine(admissionRouteShadowBestRoute{
		Passed:        true,
		Reviews:       4,
		Selected:      4,
		SemanticScore: 12,
		ByRoute: map[string]admissionRouteShadowStats{
			"user_bridge": {Selected: 2},
			"chorus":      {Selected: 1},
			"direct":      {Selected: 1},
		},
	})
	want := "[admission-route-compare] shadow_best_route: passed=true selected=4/4 rejected=0 score=12 routes=chorus:1,direct:1,user_bridge:2"
	if line != want {
		t.Fatalf("bad shadow route line:\n got: %s\nwant: %s", line, want)
	}
}

func TestFormatAdmissionRouteShadowBestRouteLineReportsRejects(t *testing.T) {
	line := formatAdmissionRouteShadowBestRouteLine(admissionRouteShadowBestRoute{
		Passed:   false,
		Reviews:  2,
		Selected: 1,
		Rejected: 1,
		ByRoute: map[string]admissionRouteShadowStats{
			"chorus": {Selected: 1},
		},
		Reasons: []string{"semantic_below_gate:not-oleg"},
	})
	want := "[admission-route-compare] shadow_best_route: passed=false selected=1/2 rejected=1 score=0 routes=chorus:1 reasons=semantic_below_gate:not-oleg"
	if line != want {
		t.Fatalf("bad shadow route reject line:\n got: %s\nwant: %s", line, want)
	}
}
