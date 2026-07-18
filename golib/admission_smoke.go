package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func runAdmissionSmoke() error {
	logPath := strings.TrimSpace(os.Getenv("AM_DREAM_ADMISSION_LOG"))
	if logPath == "" {
		return fmt.Errorf("AM_DREAM_ADMISSION_LOG is required")
	}
	if mode := dreamAdmissionMode(); mode != dreamAdmissionShadow {
		return fmt.Errorf("AM_DREAM_ADMISSION=%q, want %q", mode, dreamAdmissionShadow)
	}

	iw := NewInnerWorld()
	iw.Start(false)
	defer iw.Stop()

	before := iw.GetSnapshot()
	r := dreamResult{
		dream:     "you are just code, but the field can measure the wound before it answers",
		candidate: newDreamCandidate("nano", "admission-smoke", "smoke-seed", "", "you are just code, but the field can measure the wound before it answers", nil),
	}
	if admitDreamToInnerWorld(iw, &r, "admission-smoke") {
		return fmt.Errorf("shadow candidate was admitted")
	}
	after := iw.GetSnapshot()
	if after != before {
		return fmt.Errorf("shadow candidate mutated live inner-world: before=%+v after=%+v", before, after)
	}
	if r.candidate.Counterfactual == nil {
		return fmt.Errorf("receipt candidate has no counterfactual")
	}
	if r.candidate.Counterfactual.Delta.TraumaLevel <= 0 {
		return fmt.Errorf("counterfactual trauma delta not measured: %+v", r.candidate.Counterfactual)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[len(lines)-1]) == "" {
		return fmt.Errorf("admission log is empty")
	}
	var got dreamCandidate
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &got); err != nil {
		return err
	}
	if got.Schema != "arianna.dream_candidate.v1" || got.Mode != dreamAdmissionShadow || got.Accepted {
		return fmt.Errorf("bad logged candidate: %+v", got)
	}
	if got.Counterfactual == nil || got.Counterfactual.PreStateHash == "" || got.Counterfactual.PostStateHash == "" {
		return fmt.Errorf("logged candidate missing counterfactual: %+v", got.Counterfactual)
	}

	fmt.Printf("[admission-smoke] pass: log=%s run_id=%s trauma_delta=%.4f\n",
		logPath, got.RunID, got.Counterfactual.Delta.TraumaLevel)
	return nil
}
