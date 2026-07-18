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
	out := admissionRouteOutput{route: "qloop", text: "What remains if the field stays?", cells: []chorusCell{{text: "What remains if the field stays?", qloop: true}}, questions: 1}
	if err := recordAdmissionRouteCandidate(iw, &summary, 3, out, "qloop-test", "seed", "frag"); err != nil {
		t.Fatal(err)
	}
	if summary.Candidates != 1 || summary.ByRoute["qloop"].Produced != 1 || summary.ByRoute["qloop"].QloopQuestions != 1 {
		t.Fatalf("bad route summary: %+v", summary)
	}
}
