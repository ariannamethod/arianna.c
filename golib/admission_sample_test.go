package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAdmissionSampleBuiltinsWriteSummary(t *testing.T) {
	t.Setenv("AM_DREAM_ADMISSION", dreamAdmissionShadow)
	dir := t.TempDir()
	logPath := filepath.Join(dir, "dream-admission.jsonl")
	summaryPath := filepath.Join(dir, "dream-admission-summary.json")
	t.Setenv("AM_DREAM_ADMISSION_LOG", logPath)
	t.Setenv("AM_DREAM_ADMISSION_SUMMARY", summaryPath)

	if err := runAdmissionSample(); err != nil {
		t.Fatal(err)
	}

	var summary dreamAdmissionSampleSummary
	rawSummary, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rawSummary, &summary); err != nil {
		t.Fatal(err)
	}
	if summary.Schema != "arianna.dream_admission_sample_summary.v1" {
		t.Fatalf("bad schema: %+v", summary)
	}
	if summary.Samples != len(defaultAdmissionSamples()) {
		t.Fatalf("bad sample count: %+v", summary)
	}
	if summary.PolicyPassed == 0 || summary.PolicyFailed == 0 {
		t.Fatalf("builtin sample must cover pass and fail policy paths: %+v", summary)
	}
	if summary.ReplayFailed != 0 {
		t.Fatalf("replay failures in sampler: %+v", summary)
	}
	if summary.MaxTrauma <= 0 || summary.MaxAbsCoherence <= 0 {
		t.Fatalf("summary did not accumulate counterfactual deltas: %+v", summary)
	}

	rawLog, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(rawLog)), "\n")
	if len(lines) != summary.Samples {
		t.Fatalf("receipt count mismatch: lines=%d summary=%+v", len(lines), summary)
	}
	for i, line := range lines {
		var c dreamCandidate
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("receipt %d: %v", i+1, err)
		}
		if c.Schema != "arianna.dream_candidate.v1" || c.Mode != dreamAdmissionShadow || c.Accepted {
			t.Fatalf("bad receipt %d: %+v", i+1, c)
		}
		if c.Counterfactual == nil || c.Admission == nil {
			t.Fatalf("receipt %d missing guard data: %+v", i+1, c)
		}
	}
}

func TestAdmissionSampleLoadsPlainTextAndJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "samples.jsonl")
	receipt := newDreamCandidate("chorus", "test-trigger", "receipt-seed", "fragment", "a receipt line stays typed", nil)
	receiptRaw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	content := strings.Join([]string{
		"# comments are ignored",
		"a plain trace enters the shadow sampler",
		`{"source":"nano","trigger":"direct","seed":"json-seed","fragment":"frag","text":"json sample text"}`,
		string(receiptRaw),
		"",
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}

	samples, err := loadAdmissionSamples(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 3 {
		t.Fatalf("expected 3 samples, got %d: %+v", len(samples), samples)
	}
	if samples[0].Source != "sampler" || samples[0].Text == "" {
		t.Fatalf("plain text sample not loaded: %+v", samples[0])
	}
	if samples[1].Source != "nano" || samples[1].Trigger != "direct" || samples[1].Seed != "json-seed" || samples[1].Fragment != "frag" {
		t.Fatalf("typed sample not preserved: %+v", samples[1])
	}
	if samples[2].Source != "chorus" || samples[2].Trigger != "test-trigger" || samples[2].Seed != "receipt-seed" || samples[2].Fragment != "fragment" {
		t.Fatalf("receipt sample not preserved: %+v", samples[2])
	}
}
