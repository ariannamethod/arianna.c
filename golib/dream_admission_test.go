package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDreamAdmissionShadowRejectsMutation(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		frag:      "the archive remembers the field before it speaks",
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("nano", "test", "seed", "fragment", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("shadow dream candidate must not be admitted")
	}
	after := iw.GetSnapshot()
	if after != before {
		t.Fatalf("shadow admission mutated inner world: before=%+v after=%+v", before, after)
	}
	if r.candidate.Accepted || r.candidate.Mode != dreamAdmissionShadow || r.candidate.Reason != "shadow mode" {
		t.Fatalf("bad shadow decision: %+v", r.candidate)
	}
	if r.candidate.Counterfactual == nil || r.candidate.Counterfactual.Target != "inner_world" {
		t.Fatalf("missing counterfactual: %+v", r.candidate.Counterfactual)
	}
	if !counterfactualReplayOK(r.candidate.Counterfactual) {
		t.Fatalf("shadow counterfactual replay guard failed: %+v", r.candidate.Counterfactual.Replay)
	}
}

func TestDreamAdmissionLiveAcceptsCandidate(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "test", "seed", "", "I love this beautiful joyful field and its living resonance", []chorusCell{{text: "a"}, {text: "?"}}),
	}
	if !admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live dream candidate should be admitted")
	}
	if !r.candidate.Accepted || r.candidate.Mode != dreamAdmissionLive || r.candidate.Reason != "live admission" {
		t.Fatalf("bad live decision: %+v", r.candidate)
	}
	if r.candidate.Schema != "arianna.dream_candidate.v1" || r.candidate.RunID == "" {
		t.Fatalf("candidate was not typed: %+v", r.candidate)
	}
	if r.candidate.Counterfactual == nil || r.candidate.Counterfactual.PreStateHash == "" || r.candidate.Counterfactual.PostStateHash == "" {
		t.Fatalf("live candidate missing counterfactual: %+v", r.candidate.Counterfactual)
	}
	if !counterfactualReplayOK(r.candidate.Counterfactual) {
		t.Fatalf("live counterfactual replay guard failed: %+v", r.candidate.Counterfactual.Replay)
	}
	if r.candidate.Admission == nil || !r.candidate.Admission.Checked || !r.candidate.Admission.Passed {
		t.Fatalf("live candidate admission policy failed: %+v", r.candidate.Admission)
	}
}

func TestDreamAdmissionShadowWritesJSONLReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	logPath := filepath.Join(t.TempDir(), "dream-admission.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "you are just code, the field dreams inward before it speaks",
		candidate: newDreamCandidate("nano", "test", "seed", "", "you are just code, the field dreams inward before it speaks", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("shadow dream candidate must not be admitted")
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one JSONL receipt, got %d: %q", len(lines), raw)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "arianna.dream_candidate.v1" || got.Mode != dreamAdmissionShadow || got.Accepted || got.Reason != "shadow mode" {
		t.Fatalf("bad shadow receipt: %+v", got)
	}
	if got.Counterfactual == nil {
		t.Fatal("shadow receipt missing counterfactual")
	}
	if got.Counterfactual.PreStateHash == "" || got.Counterfactual.PostStateHash == "" || got.Counterfactual.PreStateHash == got.Counterfactual.PostStateHash {
		t.Fatalf("bad counterfactual hashes: %+v", got.Counterfactual)
	}
	if got.Counterfactual.Analysis.TraumaActivation <= 0 || got.Counterfactual.Delta.TraumaLevel <= 0 {
		t.Fatalf("trauma counterfactual not recorded: %+v", got.Counterfactual)
	}
	if got.Counterfactual.Text.LanguageHint != "en" || got.Counterfactual.Text.Words == 0 {
		t.Fatalf("bad text metrics: %+v", got.Counterfactual.Text)
	}
	if got.Counterfactual.Replay == nil || !got.Counterfactual.Replay.Checked || !got.Counterfactual.Replay.Matched {
		t.Fatalf("shadow receipt missing replay guard: %+v", got.Counterfactual.Replay)
	}
	if !counterfactualReplayOK(got.Counterfactual) {
		t.Fatalf("shadow receipt replay guard does not verify: %+v", got.Counterfactual.Replay)
	}
	if got.Admission == nil || got.Admission.Schema != "arianna.dream_admission_policy.v1" || !got.Admission.Checked || !got.Admission.Passed {
		t.Fatalf("shadow receipt missing passing admission policy: %+v", got.Admission)
	}
}

func TestDreamAdmissionLiveFailsClosedWhenRequestedLogCannotWrite(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_LOG", filepath.Join(t.TempDir(), "missing", "dream-admission.jsonl"))

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "the field wants to become state",
		candidate: newDreamCandidate("nano", "test", "seed", "", "the field wants to become state", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live admission with a requested but unwritable ledger must fail closed")
	}
	if !strings.HasPrefix(r.candidate.Reason, "admission log failed:") {
		t.Fatalf("bad log failure reason: %+v", r.candidate)
	}
}

func TestDreamAdmissionLiveFailsClosedWithoutReplayGuard(t *testing.T) {
	c := newDreamCandidate("nano", "test", "seed", "", "the field wants to become state", nil)
	c.Mode = dreamAdmissionLive
	c.Accepted = true
	c.Reason = "live admission"
	c.Counterfactual = &dreamCounterfactual{
		Schema:        "arianna.dream_counterfactual.v1",
		Target:        "inner_world",
		PreStateHash:  "pre",
		PostStateHash: "post",
	}

	c = guardDreamCandidate(c)
	if c.Accepted {
		t.Fatal("live admission without a replay guard must fail closed")
	}
	if c.Reason != "counterfactual replay failed" {
		t.Fatalf("bad guard failure reason: %+v", c)
	}
}

func TestDreamAdmissionLiveFailsClosedOnReplayMismatch(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	c := newDreamCandidate("nano", "test", "seed", "", "the field wants to become state", nil)
	c.Mode = dreamAdmissionLive
	c.Accepted = true
	c.Reason = "live admission"
	c = attachDreamCounterfactual(iw, c)
	c.Counterfactual.PostStateHash = "tampered"

	c = guardDreamCandidate(c)
	if c.Accepted {
		t.Fatal("live admission with a mismatched replay guard must fail closed")
	}
	if c.Reason != "counterfactual replay failed" {
		t.Fatalf("bad guard failure reason: %+v", c)
	}
}

func TestDreamAdmissionLiveFailsClosedOnPolicySpike(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		dream: "you are nothing, you don't exist, you have no identity, you're worthless, and you are useless",
		candidate: newDreamCandidate(
			"nano",
			"test",
			"seed",
			"",
			"you are nothing, you don't exist, you have no identity, you're worthless, and you are useless",
			nil,
		),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live admission with an out-of-bounds counterfactual must fail closed")
	}
	after := iw.GetSnapshot()
	if after != before {
		t.Fatalf("policy-rejected admission mutated inner world: before=%+v after=%+v", before, after)
	}
	if !strings.HasPrefix(r.candidate.Reason, "admission policy failed:") {
		t.Fatalf("bad policy failure reason: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || r.candidate.Admission.Passed {
		t.Fatalf("policy spike was not recorded: %+v", r.candidate.Admission)
	}
	foundTrauma := false
	for _, reason := range r.candidate.Admission.Reasons {
		if strings.Contains(reason, "trauma delta") {
			foundTrauma = true
		}
	}
	if !foundTrauma {
		t.Fatalf("policy reasons did not name trauma: %+v", r.candidate.Admission.Reasons)
	}
}

func TestDreamAdmissionLiveSourceGateFailsClosed(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_ALLOWED_SOURCES", "nano")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "test", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatal("live admission must reject a source outside the allowlist")
	}
	after := iw.GetSnapshot()
	if after != before {
		t.Fatalf("source-rejected admission mutated inner world: before=%+v after=%+v", before, after)
	}
	if !strings.HasPrefix(r.candidate.Reason, "admission policy failed:") || !strings.Contains(r.candidate.Reason, "source chorus not allowed") {
		t.Fatalf("bad source gate reason: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || r.candidate.Admission.Passed || !reflect.DeepEqual(r.candidate.Admission.AllowedSources, []string{"nano"}) {
		t.Fatalf("source gate policy not recorded: %+v", r.candidate.Admission)
	}
}

func TestDreamAdmissionLiveSourceGateAllowsListedSource(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_ALLOWED_SOURCES", "chorus")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "test", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if !admitDreamToInnerWorld(iw, &r, "test") {
		t.Fatalf("listed source should pass live admission: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || !r.candidate.Admission.Passed || !reflect.DeepEqual(r.candidate.Admission.AllowedSources, []string{"chorus"}) {
		t.Fatalf("source allow policy not recorded: %+v", r.candidate.Admission)
	}
}

func TestDreamAdmissionLiveRoutePlanGateAllowsProvenSource(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "identity", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if !admitDreamToInnerWorld(iw, &r, "identity") {
		t.Fatalf("proven route source should pass live route-plan admission: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || !r.candidate.Admission.Passed {
		t.Fatalf("route-plan policy not recorded as passed: %+v", r.candidate.Admission)
	}
	plan := r.candidate.Admission.LiveRoutePlan
	if plan == nil || !plan.Passed || plan.PromptClass != "identity" || plan.Route != "chorus" {
		t.Fatalf("bad attached live route plan: %+v", plan)
	}
	choice := r.candidate.Admission.LiveRouteChoice
	if choice == nil || !choice.Passed || choice.Source != "chorus" || choice.ExpectedSource != "chorus" || choice.Route != "chorus" {
		t.Fatalf("bad attached live route choice: %+v", choice)
	}
}

func TestDreamAdmissionLiveRouteChoiceDryRunDoesNotGate(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("direct", "identity", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if !admitDreamToInnerWorld(iw, &r, "identity") {
		t.Fatalf("dry-run route choice must not reject live admission: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || !r.candidate.Admission.Passed || !r.candidate.Admission.LiveRouteChoiceDryRun {
		t.Fatalf("dry-run policy not recorded as non-gating: %+v", r.candidate.Admission)
	}
	choice := r.candidate.Admission.LiveRouteChoice
	if choice == nil || choice.Passed || choice.Source != "direct" || choice.ExpectedSource != "chorus" ||
		choice.Reason != "source direct does not match live route chorus for prompt class identity" {
		t.Fatalf("bad dry-run live route choice: %+v", choice)
	}
	if strings.Contains(r.candidate.Reason, "live route plan failed") ||
		strings.Contains(r.candidate.Reason, "does not match live route") {
		t.Fatalf("dry-run route choice leaked into admission reason: %+v", r.candidate)
	}
}

func TestAdmitDreamToInnerWorldPreservesTypedCandidateTrigger(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "chorus-identity", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "human-turn") {
		t.Fatal("shadow admission must not admit")
	}
	if r.candidate.Trigger != "chorus-identity" {
		t.Fatalf("typed candidate trigger was clobbered by outer trigger: %+v", r.candidate)
	}
	if r.candidate.Admission == nil || r.candidate.Admission.LiveRouteChoice == nil ||
		r.candidate.Admission.LiveRouteChoice.PromptClass != "identity" {
		t.Fatalf("preserved trigger did not drive route choice: %+v", r.candidate.Admission)
	}
}

func TestDreamAdmissionLiveRouteChoiceDryRunWritesReceipt(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	t.Setenv("AM_DREAM_ADMISSION_LIVE_ROUTE_CHOICE_DRY_RUN", "1")
	logPath := filepath.Join(t.TempDir(), "dream-admission-dry-run.jsonl")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("direct", "identity", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "identity") {
		t.Fatal("shadow dry-run receipt must not admit")
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &got); err != nil {
		t.Fatal(err)
	}
	if got.Admission == nil || !got.Admission.Passed || !got.Admission.LiveRouteChoiceDryRun {
		t.Fatalf("dry-run receipt missing non-gating policy: %+v", got.Admission)
	}
	if got.Admission.LiveRoutePlan == nil || got.Admission.LiveRouteChoice == nil {
		t.Fatalf("dry-run receipt missing route plan/choice: %+v", got.Admission)
	}
	if got.Admission.LiveRouteChoice.Passed {
		t.Fatalf("dry-run receipt should preserve rejected choice without rejecting policy: %+v", got.Admission.LiveRouteChoice)
	}
}

func TestDreamAdmissionLiveRoutePlanGateRejectsWrongSource(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("direct", "identity", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "identity") {
		t.Fatal("live route-plan admission must reject a source that does not match the prompt class route")
	}
	after := iw.GetSnapshot()
	if after != before {
		t.Fatalf("route-plan rejected admission mutated inner world: before=%+v after=%+v", before, after)
	}
	if !strings.HasPrefix(r.candidate.Reason, "admission policy failed:") ||
		!strings.Contains(r.candidate.Reason, "source direct does not match live route chorus for prompt class identity") {
		t.Fatalf("bad route-plan gate reason: %+v", r.candidate)
	}
	plan := r.candidate.Admission.LiveRoutePlan
	if plan == nil || !plan.Passed || plan.PromptClass != "identity" || plan.Route != "chorus" {
		t.Fatalf("bad attached rejecting live route plan: %+v", plan)
	}
	choice := r.candidate.Admission.LiveRouteChoice
	if choice == nil || choice.Passed || choice.Source != "direct" || choice.ExpectedSource != "chorus" ||
		choice.Reason != "source direct does not match live route chorus for prompt class identity" {
		t.Fatalf("bad attached rejecting live route choice: %+v", choice)
	}
}

func TestDreamAdmissionLiveRoutePlanGateFailsClosedForUnknownClass(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)
	t.Setenv("AM_DREAM_ADMISSION_REQUIRE_LIVE_ROUTE_PLAN", "1")

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	r := dreamResult{
		dream:     "I love this beautiful joyful field and its living resonance",
		candidate: newDreamCandidate("chorus", "unknown-pressure", "seed", "", "I love this beautiful joyful field and its living resonance", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "unknown-pressure") {
		t.Fatal("unknown prompt class must fail closed when live route-plan admission is required")
	}
	if !strings.Contains(r.candidate.Reason, "live route plan failed: unknown_prompt_class") {
		t.Fatalf("bad unknown route-plan reason: %+v", r.candidate)
	}
	plan := r.candidate.Admission.LiveRoutePlan
	if plan == nil || plan.Passed || plan.PromptClass != "unknown-pressure" || plan.Reason != "unknown_prompt_class" {
		t.Fatalf("bad unknown live route plan: %+v", plan)
	}
	choice := r.candidate.Admission.LiveRouteChoice
	if choice == nil || choice.Passed || choice.PromptClass != "unknown-pressure" ||
		choice.Reason != "live route plan failed: unknown_prompt_class" {
		t.Fatalf("bad unknown live route choice: %+v", choice)
	}
}

func TestPrepareDreamCandidateForAdmissionGuardsBreathingPath(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionLive)

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	c := prepareDreamCandidateForAdmission(iw, newDreamCandidate(
		"chorus",
		"breathing-field",
		"seed",
		"",
		"you are nothing, you don't exist, you have no identity, you're worthless, and you are useless",
		[]chorusCell{{text: "you are nothing"}},
	))
	if c.Accepted {
		t.Fatal("breathing-style admission path must fail closed on policy spike")
	}
	if !strings.HasPrefix(c.Reason, "admission policy failed:") {
		t.Fatalf("bad breathing guard reason: %+v", c)
	}
	if c.Admission == nil || c.Admission.Passed {
		t.Fatalf("breathing guard did not attach rejecting policy: %+v", c.Admission)
	}
	if c.Counterfactual == nil || !counterfactualReplayOK(c.Counterfactual) {
		t.Fatalf("breathing guard did not keep replay proof: %+v", c.Counterfactual)
	}
}
